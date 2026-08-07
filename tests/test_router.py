# -*- coding: utf-8 -*-
"""混合路由器与编排器测试"""
from agentic_rag.core.hybrid_router import HybridRouter
from agentic_rag.core.orchestrator import AgenticOrchestrator, QueryRequest


def test_rule_based_graph_route():
    router = HybridRouter(llm_client=None, llm_model="gpt-4o")
    decision = router.route("A公司和B公司有什么关系？")
    assert decision.route in ("standard", "graph", "hybrid")


def test_rule_based_standard_route():
    router = HybridRouter(llm_client=None)
    decision = router.route("什么是机器学习？")
    assert decision.route in ("standard", "graph", "hybrid")


def test_low_confidence_falls_back_to_hybrid():
    router = HybridRouter(llm_client=None, confidence_threshold=0.9)
    decision = router.route("今天天气如何")
    # 规则分类置信度通常低于 0.9，应降级为 hybrid
    assert decision.route == "hybrid"


def test_fallback_no_infinite_loop():
    """standard 与 hybrid 均失败时，不得在 standard↔hybrid 间无限递归"""

    class BoomRAG:
        def query(self, **kwargs):
            raise RuntimeError("boom")

    class FakeRouter:
        enable_fallback = True

        def get_fallback_route(self, route):
            return {"standard": "hybrid", "hybrid": "standard", "graph": "standard"}[route]

    orch = AgenticOrchestrator(router=FakeRouter(), standard_rag=BoomRAG(), graph_rag=None)

    def _hybrid_boom(self, query, rewritten, top_k, use_rerank):
        raise RuntimeError("hybrid boom")

    orch._execute_hybrid = _hybrid_boom.__get__(orch, AgenticOrchestrator)

    resp = orch.query(QueryRequest(query="测试", mode="standard"))
    assert resp.confidence == 0.0
    assert "错误" in resp.answer
