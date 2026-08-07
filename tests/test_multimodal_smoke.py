"""
多模态检索冒烟测试（RAG-Anything 风格：知识图谱构建图片/表格引用 + 检索使用）
使用轻量组件（不加载本地嵌入/重排模型），验证：
1. 摄入时检测文本中图/表引用并记录位置
2. 图存储中构建 chunk -> media 引用边
3. 检索命中带引用的文本块，并能通过引用图取回媒体
4. 端到端查询（无 LLM 降级生成）不报错
"""

import os
import tempfile

from agentic_rag.core.orchestrator import AgenticOrchestrator, QueryRequest
from agentic_rag.memory.graph_store import NetworkXStore
from agentic_rag.memory.media_store import MediaRegistry
from agentic_rag.rag.graph_rag import GraphRAGEngine
from agentic_rag.rag.hybrid_retriever import HybridRetriever
from agentic_rag.rag.standard_rag import StandardRAGEngine
from agentic_rag.service import ingest_documents


def _make_orchestrator():
    gs = NetworkXStore()
    media_store = MediaRegistry()
    retriever = HybridRetriever(graph_store=gs, vector_store=None, embedder=None)
    retriever.media_store = media_store
    graph_rag = GraphRAGEngine(graph_store=gs, llm_client=None)
    standard_rag = StandardRAGEngine(retriever=retriever, llm_client=None, vlm_client=None)
    return AgenticOrchestrator(
        standard_rag=standard_rag,
        graph_rag=graph_rag,
        hybrid_retriever=retriever,
        media_store=media_store,
        llm_client=None,
        llm_model="gpt-4o-mini",
        enable_multimodal=True,
    ), gs, media_store


def test_multimodal_ingest_and_retrieve():
    orch, gs, media_store = _make_orchestrator()

    tmpdir = tempfile.mkdtemp(prefix="agr_mm_test_")
    index_path = os.path.join(tmpdir, "chunks.jsonl")
    try:
        result = ingest_documents(
            orch,
            documents=[
                {
                    "content": (
                        "2024 年公司产品销量持续增长。如图1所示，第二季度销量达到 120 万台，"
                        "创下历史新高。表2 列出了各区域的具体财务数据，其中华东地区贡献最大。"
                    ),
                    "modality": "text",
                    "metadata": {"title": "年报摘要", "source": "demo"},
                },
                {
                    "content": "公司致力于人工智能与大模型研发，员工超过 5000 人。",
                    "modality": "text",
                    "metadata": {"title": "公司介绍"},
                },
            ],
            build_graph=True,
            index_path=index_path,
        )

        assert result["status"] == "success"
        assert result["reference_count"] >= 2, f"应检测到图/表引用，实际 {result['reference_count']}"
        print("ingest:", {k: v for k, v in result.items() if k != "graph_stats"})

        # 1. 图存储中存在媒体节点与引用边
        media_nodes = gs.list_media()
        print("图存储媒体节点:", media_nodes)
        assert len(media_nodes) >= 1

        # 2. 检索命中含引用的文本块
        retrieved = orch.hybrid_retriever.retrieve(query="产品销量", top_k=5)
        hit_with_refs = [d for d in retrieved if getattr(d, "media_refs", [])]
        assert hit_with_refs, "应检索到带图/表引用的文本块"
        print("命中引用:", hit_with_refs[0].media_refs)

        # 3. 媒体检索（通过引用图扩展）
        media = orch.hybrid_retriever.retrieve_media(hit_with_refs[:2])
        assert media, "应能通过引用图检索到媒体"
        print("媒体检索:", [(m["id"], m["type"], m["page"], m["refs"]) for m in media])

        # 4. 端到端查询（无 LLM 走降级生成）
        resp = orch.query(QueryRequest(query="产品销量如何？", mode="standard", enable_multimodal=True))
        assert resp.answer
        assert resp.metadata.get("media_count", 0) >= 1 or any(
            s.get("media") for s in resp.sources
        )
        print("答案片段:", resp.answer[:100])
        print("来源含媒体:", [s.get("media", {}).get("id") for s in resp.sources if s.get("media")])

        # 5. VLM 未配置状态
        assert orch.vlm_client is None
        print("VLM 配置状态: not configured（符合预期）")
    finally:
        try:
            os.remove(index_path)
            os.rmdir(tmpdir)
        except OSError:
            pass


def test_media_registry_memory_limit_and_disk_reload():
    """验证：媒体数据超过内存上限时被卸载，但仍可从磁盘懒加载恢复"""
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="agr_media_test_")
    persist = os.path.join(tmpdir, "media.json")
    try:
        from agentic_rag.memory.media_store import MediaRegistry
        from agentic_rag.memory.multi_modal_parser import MediaAsset

        reg = MediaRegistry(persist_path=persist, auto_save=True, max_memory_bytes=100)
        # 两条各 200 字节的 base64 数据，超过 100 字节上限
        reg.add_many([
            MediaAsset(id="img_a", doc_id="doc1", type="image", page=1, label="图1",
                       data="A" * 200, metadata={}),
            MediaAsset(id="img_b", doc_id="doc1", type="image", page=2, label="图2",
                       data="B" * 200, metadata={}),
        ])
        # 内存中至少一条被卸载（data=None），但元数据仍在
        assert reg.count == 2
        a = reg._assets["img_a"]
        b = reg._assets["img_b"]
        assert a.data is None or b.data is None, "超过内存上限后应卸载部分图片数据"

        # 从磁盘懒加载恢复
        restored = reg.get("img_a")
        assert restored is not None
        assert restored.data == "A" * 200, "应从磁盘懒加载恢复图片数据"
        restored_b = reg.get("img_b")
        assert restored_b.data == "B" * 200

        # 持久化文件存在且包含完整数据
        import json
        with open(persist, encoding="utf-8") as f:
            items = json.load(f)
        assert len(items) == 2
        assert any(it["id"] == "img_a" and it["data"] == "A" * 200 for it in items)
    finally:
        try:
            os.remove(persist)
            os.rmdir(tmpdir)
        except OSError:
            pass


def test_pdf_parser_image_element_registers_media(monkeypatch):
    """验证：Unstructured 返回 Image 元素（带 image_base64）时，注册媒体资产并建立引用"""
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="agr_pdf_img_")
    pdf_file = os.path.join(tmpdir, "demo.pdf")
    with open(pdf_file, "wb") as f:
        f.write(b"%PDF-1.4 fake")

    from agentic_rag.memory.multi_modal_parser import PDFParser
    parser = PDFParser(llm_client=None)

    # 伪造：跳过 PyMuPDF 抽取，返回空媒体
    monkeypatch.setattr(parser, "_extract_media", lambda *a, **k: ([], {}))

    # 伪造 Unstructured 元素：一个文本 + 一个 Image（带 base64）
    class FakeMeta:
        image_base64 = "QUJDREVGRw=="  # "ABCDEFG"

    class FakeImage:
        def __init__(self):
            self.metadata = FakeMeta()

        def __str__(self):
            return ""

    class FakeText:
        def __init__(self):
            self.metadata = None

        def __str__(self):
            return "如图1所示，销量增长。Table 2 列出财务数据。"

    fake_elements = [FakeText(), FakeImage()]

    # 在 sys.modules 注入假 unstructured.partition.pdf 模块（真实模块因缺依赖无法导入）
    import sys
    import types
    fake_pkg = types.ModuleType("unstructured.partition.pdf")
    fake_pkg.partition_pdf = lambda **k: fake_elements
    monkeypatch.setitem(sys.modules, "unstructured.partition.pdf", fake_pkg)

    parsed = parser.parse(pdf_file, "doc_pdf_img", {"title": "demo"})
    # 注册了新的图片资产（来自 Image 元素 base64）
    assert any(m.type == "image" and m.data == "QUJDREVGRw==" for m in parsed.media), "应从 Image 元素注册媒体资产"
    # Image chunk 有引用
    img_chunk = [c for c in parsed.chunks if c.modality == "image"]
    assert img_chunk, "应存在 image 模态的分块"
    assert img_chunk[0].media_refs, "Image 分块应建立媒体引用"
    # 文本 chunk 检测到图/表引用
    text_chunk = [c for c in parsed.chunks if c.modality == "text"]
    assert text_chunk and len(text_chunk[0].media_refs) >= 2
    try:
        os.remove(pdf_file)
        os.rmdir(tmpdir)
    except OSError:
        pass


if __name__ == "__main__":
    test_multimodal_ingest_and_retrieve()
    test_media_registry_memory_limit_and_disk_reload()
    print("✅ 多模态检索冒烟测试通过")
