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
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.embedder = embedder
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.top_k_rerank = top_k_rerank

    def query(
        self,
        query_text: str,
        top_k: int = 20,
        use_rerank: bool = True,
        system_prompt: Optional[str] = None,
        rewritten_query: Optional[Any] = None,
    ) -> RAGResult:
        """
        执行标准 RAG 查询

        Args:
            query_text: 用户查询
            top_k: 初始检索数量
            use_rerank: 是否使用重排序
            system_prompt: 自定义系统提示
            rewritten_query: 重写后的查询（variants/sub_queries/hyde_answer 会参与扩展检索）

        Returns:
            RAGResult: 包含答案和来源
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

        # 3. 构建上下文
        context = self._build_context(retrieved_docs)

        # 4. 生成答案
        answer = self._generate_answer(query_text, context, system_prompt)

        elapsed = (time.time() - start_time) * 1000

        return RAGResult(
            answer=answer,
            sources=[
                {
                    "doc_id": d.doc_id,
                    "content": d.content[:200],
                    "score": d.score,
                    "modality": getattr(d, "modality", "text"),
                }
                for d in retrieved_docs[:self.top_k_rerank]
            ],
            route="standard",
            confidence=min(1.0, sum(d.score for d in retrieved_docs[:3]) / 3) if retrieved_docs else 0.0,
            latency_ms=elapsed,
        )

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