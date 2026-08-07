"""
文档摄入服务模块
================
将多模态文档解析、分块、嵌入、入库、建图的完整流程封装为可复用服务。
供 FastAPI 路由与 AgenticRAG 高层客户端共用。
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from agentic_rag.rag.hybrid_retriever import ScoredDocument


def ingest_documents(
    orchestrator,
    documents: List[Dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    build_graph: bool = True,
    index_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    摄入多模态文档

    Args:
        orchestrator: AgenticOrchestrator 实例
        documents: 文档列表，每项形如
            {"content": str, "modality": "text|image|table|pdf", "metadata": dict, "collection": str}
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        build_graph: 是否构建知识图谱（含图片/表格引用图）
        index_path: BM25 索引持久化路径（默认 data/index/chunks.jsonl，测试可传临时路径避免污染）

    Returns:
        dict: {"status", "doc_count", "chunk_count", "graph_stats", "media_count", "message"}
    """
    from agentic_rag.memory.multi_modal_parser import MultiModalParser
    from agentic_rag.memory.vector_store import VectorRecord
    from agentic_rag.processing.chunker import get_chunker

    parser = MultiModalParser(
        llm_client=getattr(orchestrator, "llm_client", None),
        llm_model=getattr(orchestrator, "llm_model", None) or "gpt-4o",
        vlm_client=getattr(orchestrator, "vlm_client", None),
        vlm_model=getattr(orchestrator, "vlm_model", None),
    )
    chunker = get_chunker("recursive", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    hybrid_retriever = getattr(orchestrator, "hybrid_retriever", None)
    graph_rag = getattr(orchestrator, "graph_rag", None)
    media_store = getattr(orchestrator, "media_store", None)

    total_chunks = 0
    total_docs = len(documents)
    total_media = 0
    total_refs = 0
    graph_stats = None
    chunk_seq = 0  # 全局分块序号，保证 chunk_id 跨文档/跨分块唯一
    # 收集所有分块，用于构建 BM25 关键词索引（不依赖嵌入模型）
    index_chunks: List[ScoredDocument] = []

    for doc_item in documents:
        # 1. 解析文档（文本/图片/表格/PDF -> 统一 ParsedDocument）
        parsed = parser.parse(
            content=doc_item.get("content", ""),
            modality=doc_item.get("modality", "text"),
            metadata=doc_item.get("metadata", {}),
        )

        # 1.1 注册媒体资产（图片/表格）
        if parsed.media:
            total_media += len(parsed.media)
            if media_store:
                media_store.add_many(parsed.media)

        # 2. 分块
        for parsed_chunk in parsed.chunks:
            # 引用位置（文本块 → 图片/表格）：分块后对每个 chunk 重新检测，保证 offset 精确
            from agentic_rag.memory.multi_modal_parser import detect_media_refs
            media_index = {m.label: m for m in (parsed.media or []) if m.label}
            chunk_page = parsed_chunk.metadata.get("page", 1)

            chunks = chunker.chunk(
                text=parsed_chunk.content,
                doc_id=parsed.doc_id,
                metadata={
                    **doc_item.get("metadata", {}),
                    "modality": doc_item.get("modality", "text"),
                    "page": chunk_page,
                    "media_refs": [],
                },
            )
            total_chunks += len(chunks)

            # 重写 chunk_id：分块器对每个 parsed_chunk 都从 _0000 编号，
            # 若不处理，同一文档多个解析块会产生重复 ID，导致向量库相互覆盖
            for chunk in chunks:
                chunk.chunk_id = f"{parsed.doc_id}_chunk_{chunk_seq:04d}"
                chunk_seq += 1

                # 重新检测本分块内的图/表引用位置
                chunk_refs = detect_media_refs(
                    chunk.content, parsed.doc_id, page=chunk_page, media_index=media_index
                )
                ref_dicts = [{
                    "media_id": r.media_id, "media_type": r.media_type,
                    "label": r.label, "page": r.page, "offset": r.offset,
                } for r in chunk_refs]
                chunk.metadata["media_refs"] = ref_dicts
                total_refs += len(ref_dicts)

                # 4. 收集分块用于 BM25 关键词索引（任何情况下都可用）
                index_chunks.append(ScoredDocument(
                    doc_id=chunk.chunk_id,
                    content=chunk.content,
                    score=1.0,
                    metadata=chunk.metadata,
                    modality=doc_item.get("modality", "text"),
                    media_refs=ref_dicts,
                ))

                # 5. 构建媒体引用图（文本块 → 图片/表格，记录引用位置）
                if build_graph and graph_rag and graph_rag.graph_store and chunk_refs:
                    gs = graph_rag.graph_store
                    try:
                        gs.add_chunk_node(chunk.chunk_id, parsed.doc_id, chunk.content[:100])
                        for ref in chunk_refs:
                            gs.add_reference(
                                chunk_id=chunk.chunk_id,
                                media_id=ref.media_id,
                                label=ref.label,
                                page=ref.page,
                                offset=ref.offset,
                                media_type=ref.media_type,
                            )
                    except Exception as e:
                        logger.warning(f"媒体引用图构建失败: {e}")

            # 3. 生成嵌入并写入向量库（批量，需要嵌入模型可用）
            if hybrid_retriever and hybrid_retriever.embedder and hybrid_retriever.vector_store:
                texts = [c.content for c in chunks]
                embeddings = hybrid_retriever.embedder.embed(texts)
                records = []
                for chunk, embedding in zip(chunks, embeddings):
                    records.append(VectorRecord(
                        id=chunk.chunk_id,
                        vector=embedding,
                        payload={
                            "content": chunk.content,
                            "doc_id": chunk.doc_id,
                            "modality": doc_item.get("modality", "text"),
                            "media_refs": chunk.metadata.get("media_refs", []),
                            **chunk.metadata,
                        },
                    ))
                hybrid_retriever.vector_store.add(records)

        # 6. 媒体节点入库（图片/表格资产节点）
        if build_graph and graph_rag and graph_rag.graph_store and parsed.media:
            gs = graph_rag.graph_store
            try:
                for m in parsed.media:
                    gs.add_media_node(
                        media_id=m.id,
                        media_type=m.type,
                        doc_id=m.doc_id,
                        page=m.page,
                        label=m.label,
                        caption=m.caption,
                        properties={"source": m.metadata.get("source", "")},
                    )
            except Exception as e:
                logger.warning(f"媒体节点入库失败: {e}")

        # 7. 构建实体知识图谱（可选）——使用该文档全部解析块，聚合各次构建统计
        if build_graph and graph_rag and graph_rag.graph_store:
            doc_text = "\n".join(c.content for c in parsed.chunks if c.content)
            stats = graph_rag.build_graph_from_documents([
                {"id": parsed.doc_id, "content": doc_text}
            ])
            if graph_stats is None:
                graph_stats = dict(stats)
            else:
                for key in ("entities", "relations", "documents", "communities"):
                    graph_stats[key] = graph_stats.get(key, 0) + stats.get(key, 0)

    # 8. 构建 BM25 关键词索引（保证无嵌入模型时关键词检索仍可用）
    if hybrid_retriever and index_chunks:
        hybrid_retriever.build_index(index_chunks, append=True)
        # 持久化分块到磁盘，供新进程/重启后重建索引
        _persist_chunks(index_chunks, path=index_path)
        logger.info(f"BM25 索引构建完成: {len(index_chunks)} 个分块")

    result = {
        "status": "success",
        "doc_count": total_docs,
        "chunk_count": total_chunks,
        "media_count": total_media,
        "reference_count": total_refs,
        "graph_stats": graph_stats,
        "message": f"成功摄入 {total_docs} 篇文档，{total_chunks} 个分块，{total_media} 个媒体资产，{total_refs} 处图/表引用",
    }
    logger.info(result["message"])
    return result


# ── 索引持久化 ──

DEFAULT_INDEX_PATH = "data/index/chunks.jsonl"


def _persist_chunks(chunks: List[ScoredDocument], path: Optional[str] = None) -> str:
    """将分块追加持久化到 JSONL 文件（用于新进程重建 BM25 索引）"""
    import json
    import os

    path = path or DEFAULT_INDEX_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "doc_id": c.doc_id,
                "content": c.content,
                "metadata": c.metadata,
                "modality": c.modality,
                "media_refs": getattr(c, "media_refs", []) or c.metadata.get("media_refs", []),
            }, ensure_ascii=False) + "\n")
    logger.info(f"分块已持久化: {len(chunks)} 条 -> {path}")
    return path
