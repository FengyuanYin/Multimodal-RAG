"""
标准 RAG 引擎模块
================
实现常规的检索增强生成流程：查询 → 检索 → 重排序 → 生成。
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger

from agentic_rag.processing.reranker import ScoredDocument


@dataclass
class RAGResult:
    """RAG 结果"""
    answer: str
    sources: List[dict] = field(default_factory=list)
    route: str = "standard"
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class StandardRAGEngine:
    """
    标准 RAG 引擎

    流程：
    1. 查询重写（可选）
    2. 混合检索（向量 + 关键词）
    3. 重排序
    4. 答案生成
    """

    def __init__(
        self,
        retriever=None,
        reranker=None,
        embedder=None,
        llm_client=None,
        llm_model: str = "gpt-4o-mini",
        top_k_rerank: int = 5,
        vlm_client=None,
        vlm_model: Optional[str] = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.embedder = embedder
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.top_k_rerank = top_k_rerank
        self.vlm_client = vlm_client
        self.vlm_model = vlm_model

    def query(
        self,
        query_text: str,
        top_k: int = 20,
        use_rerank: bool = True,
        system_prompt: Optional[str] = None,
        rewritten_query: Optional[Any] = None,
        enable_multimodal: bool = False,
    ) -> RAGResult:
        """
        执行标准 RAG 查询

        Args:
            query_text: 用户查询
            top_k: 初始检索数量
            use_rerank: 是否使用重排序
            system_prompt: 自定义系统提示
            rewritten_query: 重写后的查询（variants/sub_queries/hyde_answer 会参与扩展检索）
            enable_multimodal: 是否启用多模态检索（通过引用图扩展图片/表格，RAG-Anything 风格）

        Returns:
            RAGResult: 包含答案和来源（启用多模态时 sources 会附加 media 项）
        """
        import time
        start_time = time.time()

        # 1. 检索（有重写结果时使用多路扩展检索）
        has_rewrite = bool(
            rewritten_query
            and (rewritten_query.variants or rewritten_query.sub_queries or rewritten_query.hyde_answer)
        )
        if has_rewrite:
            retrieved_docs = self.retriever.retrieve_with_rewrite(
                query_text, rewritten_query, top_k=top_k
            )
        else:
            retrieved_docs = self.retriever.retrieve(
                query=query_text,
                top_k=top_k,
                use_vector=True,
                use_keyword=True,
                use_graph=False,
            )

        if not retrieved_docs:
            logger.warning("检索结果为空")
            return RAGResult(
                answer="抱歉，我没有找到相关信息。",
                route="standard",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # 2. 重排序
        if use_rerank and self.reranker:
            scored_docs = [
                ScoredDocument(
                    doc_id=d.doc_id,
                    content=d.content,
                    score=d.score,
                    metadata=d.metadata,
                    modality=d.modality,
                )
                for d in retrieved_docs
            ]
            reranked = self.reranker.rerank(query_text, scored_docs, top_k=self.top_k_rerank)
            retrieved_docs = reranked

        # 2.5 多模态检索扩展：通过引用图找到关联图片/表格（RAG-Anything 风格）
        media_items = []
        if enable_multimodal and self.retriever and hasattr(self.retriever, "retrieve_media"):
            try:
                media_items = self.retriever.retrieve_media(retrieved_docs, include_data=True)
            except Exception as e:
                logger.warning(f"多模态媒体检索失败: {e}")

        # 3. 构建上下文（含媒体：图片走 VLM 描述，表格直接文本）
        context = self._build_context(retrieved_docs)
        media_context = self._build_media_context(media_items)
        if media_context:
            context = context + "\n\n" + media_context

        # 4. 生成答案
        answer = self._generate_answer(query_text, context, system_prompt)

        elapsed = (time.time() - start_time) * 1000

        sources = [
            {
                "doc_id": d.doc_id,
                "content": d.content[:200],
                "score": d.score,
                "modality": getattr(d, "modality", "text"),
                "media_refs": getattr(d, "media_refs", []) or d.metadata.get("media_refs", []),
            }
            for d in retrieved_docs[:self.top_k_rerank]
        ]
        # 媒体资产作为独立来源项
        for m in media_items[:self.top_k_rerank]:
            sources.append({
                "doc_id": m.get("doc_id", ""),
                "content": (m.get("caption") or f"[{m.get('label', m.get('id'))} {m.get('type')}]")[:200],
                "score": 1.0,
                "modality": m.get("type", "image"),
                "media": m,
            })

        return RAGResult(
            answer=answer,
            sources=sources,
            route="standard",
            confidence=min(1.0, sum(d.score for d in retrieved_docs[:3]) / 3) if retrieved_docs else 0.0,
            latency_ms=elapsed,
            metadata={"media_count": len(media_items)} if media_items else {},
        )

    def _build_media_context(self, media_items: list) -> str:
        """把关联媒体转换为 LLM 上下文：图片用 VLM 描述，表格直接给出文本表示"""
        if not media_items:
            return ""
        parts = []
        for i, m in enumerate(media_items[:8]):
            mtype = m.get("type", "image")
            label = m.get("label") or m.get("id", "")
            page = m.get("page", 1)
            if mtype == "image":
                data = m.get("data")
                if data and self.vlm_client:
                    desc = self._describe_image_with_vlm(data)
                    parts.append(f"[图片 {label}（第{page}页）]\n{desc}")
                else:
                    parts.append(f"[图片 {label}（第{page}页）]（未配置 VLM，无法生成描述；可查看引用片段）")
            elif mtype == "table":
                caption = m.get("caption") or m.get("data") or ""
                parts.append(f"[表格 {label}（第{page}页）]\n{caption[:1000]}")
            else:
                parts.append(f"[{mtype} {label}（第{page}页）]")
        return "【关联图片/表格】\n" + "\n\n".join(parts)

    def _describe_image_with_vlm(self, data: str) -> str:
        """调用 VLM 描述图片（data 为 base64）"""
        try:
            import base64 as _b64
            # 兼容 data URL 与纯 base64
            if data.startswith("data:"):
                b64 = data.split(",", 1)[1]
            else:
                b64 = data
            response = self.vlm_client.chat.completions.create(
                model=self.vlm_model or self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请用中文简要描述这张图片的核心内容（3-5 句话），说明它可能用于回答什么问题。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"VLM 描述图片失败: {e}")
            return "（VLM 描述失败）"

    def _build_context(self, documents: list) -> str:
        """构建 LLM 上下文"""
        parts = []
        for i, doc in enumerate(documents):
            content = doc.content if hasattr(doc, 'content') else getattr(doc, 'text', str(doc))
            parts.append(f"[文档 {i+1}]\n{content[:1000]}")
        return "\n\n".join(parts)

    def _generate_answer(self, query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """使用 LLM 生成答案"""
        if not self.llm_client:
            return self._fallback_generate(query, context)

        system = system_prompt or (
            "你是一个智能问答助手。请基于提供的参考文档，准确、简洁地回答用户问题。\n"
            "要求：\n"
            "1. 只使用参考文档中的信息，不要编造\n"
            "2. 如果文档信息不足，明确说明\n"
            "3. 引用相关文档编号\n"
            "4. 回答使用中文"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"参考文档：\n{context}\n\n用户问题：{query}"},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return self._fallback_generate(query, context)

    def _fallback_generate(self, query: str, context: str) -> str:
        """降级生成方案（未配置 LLM 时）：返回检索到的相关文档片段"""
        if not context.strip():
            return "抱歉，没有找到相关信息。"

        # 按文档分块截取前几段，标注来源
        parts = context.split("\n\n")
        selected = parts[:6]
        snippet = "\n\n".join(selected)
        return (
            "以下是从知识库中检索到的相关文档片段"
            "（当前未配置 LLM，无法生成综合答案，请参考原文）：\n\n"
            f"{snippet[:2000]}"
        )