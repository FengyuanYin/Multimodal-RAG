"""
Agentic 编排器模块
================
协调整个问答流程：意图识别 → 路由决策 → 查询重写 → 混合检索 → 重排序 → 答案生成。
支持多步推理、工具调用、自我反思和记忆管理。
"""

from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from loguru import logger
import time
import json


@dataclass
class QueryRequest:
    """查询请求"""
    query: str
    conversation_id: Optional[str] = None
    history: Optional[List[dict]] = None
    mode: Literal["auto", "standard", "graph", "hybrid"] = "auto"
    top_k: int = 5
    rerank: bool = True
    stream: bool = False


@dataclass
class QueryResponse:
    """查询响应"""
    answer: str
    route: str = "auto"
    confidence: float = 0.0
    sources: List[dict] = field(default_factory=list)
    conversation_id: Optional[str] = None
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class AgenticOrchestrator:
    """
    Agentic 编排器

    核心能力：
    1. 意图识别与路由决策
    2. 查询重写与优化
    3. 多路径检索协调
    4. 结果融合与重排序
    5. 答案生成与验证
    6. 自我反思与修正
    7. 对话历史管理
    """

    def __init__(
        self,
        router=None,
        query_rewriter=None,
        standard_rag=None,
        graph_rag=None,
        hybrid_retriever=None,
        reranker=None,
        llm_client=None,
        llm_model: str = "gpt-4o-mini",
    ):
        self.router = router
        self.query_rewriter = query_rewriter
        self.standard_rag = standard_rag
        self.graph_rag = graph_rag
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.llm_model = llm_model

        # 对话记忆
        self.conversations: Dict[str, List[dict]] = {}

    def query(self, request: QueryRequest) -> QueryResponse:
        """
        执行完整的 Agentic RAG 查询流程

        Args:
            request: 查询请求

        Returns:
            QueryResponse: 查询响应
        """
        start_time = time.time()
        conv_id = request.conversation_id or f"conv_{int(start_time)}"

        logger.info(f"开始处理查询: {request.query[:50]}... (mode={request.mode})")

        # 1. 获取对话历史（外部传入的 history 优先，其次使用内部会话记忆）
        internal_history = self._get_history(conv_id)
        history = request.history if request.history else internal_history

        # 2. 路由决策
        route_decision = self._decide_route(request, history)
        logger.info(f"路由决策: {route_decision.route} (置信度: {route_decision.confidence:.2f})")

        # 3. 查询重写
        rewritten = self._rewrite_query(request.query, route_decision.route)
        if rewritten and rewritten.strategy != "none":
            logger.info(f"查询重写: strategy={rewritten.strategy}")

        # 4. 执行检索与生成
        result = self._execute_route(
            query=request.query,
            rewritten=rewritten,
            route=route_decision.route,
            top_k=request.top_k * 4,  # 初始检索更多
            use_rerank=request.rerank,
        )

        # 5. 答案验证与自我反思（可选）
        if self.llm_client and result.confidence < 0.3:
            logger.info(f"答案置信度较低 ({result.confidence:.2f})，尝试修正")
            result = self._refine_answer(request.query, result)

        # 6. 保存对话历史
        self._save_history(conv_id, request.query, result.answer)

        elapsed = (time.time() - start_time) * 1000
        result.conversation_id = conv_id
        result.latency_ms = elapsed

        logger.info(f"查询完成: {elapsed:.0f}ms, route={result.route}, confidence={result.confidence:.2f}")
        return result

    def _decide_route(self, request: QueryRequest, history: Optional[list]) -> 'RouteDecision':
        """路由决策"""
        from agentic_rag.core.hybrid_router import RouteDecision

        # 如果用户指定了模式，直接使用
        if request.mode != "auto":
            return RouteDecision(
                route=request.mode,
                confidence=1.0,
                reasoning=f"用户指定模式: {request.mode}",
            )

        # 使用混合路由器
        if self.router:
            return self.router.route(request.query, history)

        # 默认 hybrid
        return RouteDecision(route="hybrid", confidence=0.5, reasoning="默认路由")

    def _rewrite_query(self, query: str, route: str) -> Optional[Any]:
        """查询重写"""
        if not self.query_rewriter:
            return None

        # GraphRAG 路径侧重分解，Standard 侧重扩展
        if route == "graph":
            return self.query_rewriter.rewrite(query, strategy="decomposition")
        elif route == "standard":
            return self.query_rewriter.rewrite(query, strategy="expansion")
        else:
            return self.query_rewriter.rewrite(query, strategy="auto")

    def _execute_route(
        self,
        query: str,
        rewritten: Optional[Any],
        route: str,
        top_k: int = 20,
        use_rerank: bool = True,
        attempted: Optional[set] = None,
    ) -> QueryResponse:
        """执行路由对应的检索与生成

        Args:
            attempted: 已尝试过的路由集合，防止 Fallback 在 standard↔hybrid 间死循环
        """
        attempted = attempted or set()
        attempted.add(route)
        try:
            if route == "standard":
                return self._execute_standard(query, rewritten, top_k, use_rerank)
            elif route == "graph":
                return self._execute_graph(query, rewritten, top_k, use_rerank)
            else:  # hybrid
                return self._execute_hybrid(query, rewritten, top_k, use_rerank)
        except Exception as e:
            logger.error(f"{route} 路由执行失败: {e}")
            if self.router and getattr(self.router, 'enable_fallback', False):
                fallback_route = self.router.get_fallback_route(route)
                if fallback_route in attempted:
                    logger.error(f"Fallback 路由 {fallback_route} 已尝试过，停止降级")
                    return QueryResponse(
                        answer=f"处理查询时发生错误: {str(e)}",
                        route=route,
                        confidence=0.0,
                    )
                logger.info(f"Fallback 到 {fallback_route}")
                return self._execute_route(query, rewritten, fallback_route, top_k, use_rerank, attempted)
            return QueryResponse(
                answer=f"处理查询时发生错误: {str(e)}",
                route=route,
                confidence=0.0,
            )

    def _execute_standard(self, query: str, rewritten: Optional[Any],
                          top_k: int, use_rerank: bool) -> QueryResponse:
        """执行标准 RAG"""
        if self.standard_rag:
            result = self.standard_rag.query(
                query_text=query,
                top_k=top_k,
                use_rerank=use_rerank,
                rewritten_query=rewritten,
            )
            return QueryResponse(
                answer=result.answer,
                route="standard",
                confidence=result.confidence,
                sources=result.sources,
                metadata={"latency_ms": result.latency_ms},
            )
        return QueryResponse(answer="标准 RAG 引擎未配置", route="standard", confidence=0.0)

    def _execute_graph(self, query: str, rewritten: Optional[Any],
                       top_k: int, use_rerank: bool) -> QueryResponse:
        """执行 GraphRAG"""
        if self.graph_rag:
            result = self.graph_rag.query(
                query_text=query,
                top_k=top_k,
                use_community=True,
            )
            return QueryResponse(
                answer=result.answer,
                route="graph",
                confidence=result.confidence,
                sources=result.sources,
                metadata={
                    "entities": result.entities,
                    "relations": result.relations,
                    "latency_ms": result.latency_ms,
                },
            )
        return QueryResponse(answer="GraphRAG 引擎未配置", route="graph", confidence=0.0)

    def _execute_hybrid(self, query: str, rewritten: Optional[Any],
                        top_k: int, use_rerank: bool) -> QueryResponse:
        """
        执行混合路由

        混合策略：
        1. 同时使用标准 RAG 和 GraphRAG
        2. 结果融合
        3. 综合生成答案
        """
        # 并行获取两种结果
        standard_result = None
        graph_result = None

        if self.standard_rag:
            try:
                standard_result = self.standard_rag.query(
                    query_text=query,
                    top_k=top_k,
                    use_rerank=use_rerank,
                    rewritten_query=rewritten,
                )
            except Exception as e:
                logger.warning(f"混合模式 - 标准 RAG 失败: {e}")

        if self.graph_rag:
            try:
                graph_result = self.graph_rag.query(
                    query_text=query,
                    top_k=top_k,
                    use_community=True,
                )
            except Exception as e:
                logger.warning(f"混合模式 - GraphRAG 失败: {e}")

        # 融合结果
        if standard_result and graph_result:
            # 使用 LLM 融合两个结果
            answer = self._fuse_answers(query, standard_result.answer, graph_result.answer)
            sources = (standard_result.sources or []) + (graph_result.sources or [])
            confidence = max(standard_result.confidence, graph_result.confidence)

            return QueryResponse(
                answer=answer,
                route="hybrid",
                confidence=confidence,
                sources=sources[:5],
                metadata={
                    "standard_confidence": standard_result.confidence,
                    "graph_confidence": graph_result.confidence,
                },
            )
        elif standard_result:
            return QueryResponse(
                answer=standard_result.answer,
                route="hybrid",
                confidence=standard_result.confidence,
                sources=standard_result.sources,
            )
        elif graph_result:
            return QueryResponse(
                answer=graph_result.answer,
                route="hybrid",
                confidence=graph_result.confidence,
                sources=graph_result.sources,
            )
        else:
            return QueryResponse(
                answer="抱歉，无法获取相关信息。",
                route="hybrid",
                confidence=0.0,
            )

    def _fuse_answers(self, query: str, standard_answer: str, graph_answer: str) -> str:
        """融合标准 RAG 和 GraphRAG 的答案"""
        if not self.llm_client:
            return f"{standard_answer}\n\n---\n\n{graph_answer}"

        system = (
            "你是一个答案融合专家。给定用户问题和两个来源的答案，"
            "将它们融合成一个连贯、完整、准确的回答。\n"
            "要求：\n"
            "1. 整合两个答案中的信息，消除矛盾\n"
            "2. 保持逻辑连贯性\n"
            "3. 标注信息来源（文档/知识图谱）\n"
            "4. 使用中文回答"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            f"用户问题：{query}\n\n"
                            f"【文档检索答案】\n{standard_answer}\n\n"
                            f"【知识图谱答案】\n{graph_answer}\n\n"
                            f"请融合以上两个答案。"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"答案融合失败: {e}")
            return f"{standard_answer}\n\n---\n\n{graph_answer}"

    def _refine_answer(self, query: str, result: QueryResponse) -> QueryResponse:
        """答案修正与自我反思"""
        if not self.llm_client:
            return result

        system = (
            "你是一个答案质量评审员。检查以下答案是否准确、完整地回答了用户问题。\n"
            "如果答案不完整或不准确，请给出改进版本。\n"
            "如果答案已经很好，请回复 'OK'。"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            f"用户问题：{query}\n\n"
                            f"当前答案：{result.answer}\n\n"
                            f"请评审并改进。"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            refined = response.choices[0].message.content
            if refined.strip() != "OK":
                result.answer = refined
                result.metadata["refined"] = True
        except Exception as e:
            logger.warning(f"答案修正失败: {e}")

        return result

    def _get_history(self, conversation_id: str) -> Optional[List[dict]]:
        """获取对话历史"""
        return self.conversations.get(conversation_id)

    def _save_history(self, conversation_id: str, query: str, answer: str):
        """保存对话历史"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append({
            "query": query,
            "answer": answer,
            "timestamp": time.time(),
        })
        # 只保留最近 20 轮
        if len(self.conversations[conversation_id]) > 20:
            self.conversations[conversation_id] = self.conversations[conversation_id][-20:]

    def clear_history(self, conversation_id: Optional[str] = None):
        """清除对话历史"""
        if conversation_id:
            self.conversations.pop(conversation_id, None)
        else:
            self.conversations.clear()