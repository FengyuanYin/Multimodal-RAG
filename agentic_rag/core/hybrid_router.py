"""
混合路由器模块
============
根据用户提问自适应选择 RAG 路径：Standard RAG、GraphRAG 或 Hybrid。
使用 LLM 进行意图分类，支持置信度阈值和 Fallback 机制。
"""

from typing import Optional, Literal
from dataclasses import dataclass, field
from loguru import logger
import json


RouteType = Literal["standard", "graph", "hybrid"]


@dataclass
class RouteDecision:
    """路由决策结果"""
    route: RouteType
    confidence: float
    reasoning: str = ""
    sub_route: Optional[str] = None  # 子路径说明


class IntentClassifier:
    """
    意图分类器
    基于 LLM 对用户查询进行分类，决定使用哪种检索路径。
    """

    # 分类提示模板
    CLASSIFICATION_PROMPT = """你是一个智能路由分析器。分析用户查询，判断最适合的检索策略。

分类规则：
- standard: 事实性查询、摘要查询、定义查询、简单问答
  - 示例："2024年GDP是多少？"、"这篇文章讲了什么？"、"什么是机器学习？"
- graph: 关系性查询、多跳推理、比较性查询、路径查询
  - 示例："A公司和B公司有什么关系？"、"A的CEO曾在哪所学校就读？"、"产品X和Y的优缺点"
- hybrid: 复杂查询、模糊查询、需要多源信息的查询
  - 示例："分析AI行业2024年的发展趋势"、"对比不同方案的优劣并给出建议"

用户查询：{query}

请以 JSON 格式输出（不要包含其他内容）：
{{
    "route": "standard|graph|hybrid",
    "confidence": 0.0-1.0,
    "reasoning": "简要说明分类理由"
}}
"""

    def __init__(self, llm_client=None, confidence_threshold: float = 0.6,
                 llm_model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.llm_model = llm_model

    def classify(self, query: str) -> RouteDecision:
        """对查询进行分类"""
        if self.llm_client:
            return self._classify_with_llm(query)
        return self._classify_with_rules(query)

    def _classify_with_llm(self, query: str) -> RouteDecision:
        """使用 LLM 进行分类"""
        try:
            prompt = self.CLASSIFICATION_PROMPT.format(query=query)
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            decision = RouteDecision(
                route=result.get("route", "hybrid"),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
            )
            logger.info(f"LLM 路由分类: {decision.route} (置信度: {decision.confidence:.2f})")
            return decision
        except Exception as e:
            logger.warning(f"LLM 分类失败: {e}，降级为规则分类")
            return self._classify_with_rules(query)

    def _classify_with_rules(self, query: str) -> RouteDecision:
        """使用规则进行分类（降级方案）"""
        query_lower = query.lower()

        # GraphRAG 关键词
        graph_keywords = [
            "关系", "关联", "联系", "路径", "网络", "结构",
            "比较", "对比", "区别", "异同",
            "谁", "哪个", "哪些", "如何影响",
            "relationship", "connection", "compare", "contrast",
            "path", "network", "hierarchy",
        ]

        # Standard RAG 关键词
        standard_keywords = [
            "是什么", "什么是", "定义", "解释", "概述",
            "总结", "摘要", "列举", "列出",
            "what is", "define", "explain", "summarize",
            "list", "example",
        ]

        # 计算匹配分数
        graph_score = sum(1 for kw in graph_keywords if kw in query_lower)
        standard_score = sum(1 for kw in standard_keywords if kw in query_lower)

        # 判断查询长度和复杂度
        is_complex = len(query) > 30 and ("?" in query or "？" in query or "分析" in query_lower)

        if graph_score > standard_score and graph_score >= 1:
            return RouteDecision(
                route="graph",
                confidence=min(0.5 + graph_score * 0.15, 0.95),
                reasoning=f"检测到 {graph_score} 个图查询关键词",
            )
        elif standard_score > graph_score and standard_score >= 1:
            return RouteDecision(
                route="standard",
                confidence=min(0.5 + standard_score * 0.15, 0.95),
                reasoning=f"检测到 {standard_score} 个标准查询关键词",
            )
        elif is_complex:
            return RouteDecision(
                route="hybrid",
                confidence=0.7,
                reasoning="复杂查询，需要混合检索",
            )
        else:
            # 默认使用 hybrid
            return RouteDecision(
                route="hybrid",
                confidence=0.5,
                reasoning="无法明确分类，使用混合策略",
            )


class HybridRouter:
    """
    混合路由器
    集成意图分类和路由决策，支持 Fallback 机制。
    """

    def __init__(
        self,
        llm_client=None,
        confidence_threshold: float = 0.6,
        enable_fallback: bool = True,
        llm_model: str = "gpt-4o",
    ):
        self.classifier = IntentClassifier(llm_client, confidence_threshold, llm_model)
        self.confidence_threshold = confidence_threshold
        self.enable_fallback = enable_fallback
        self.llm_model = llm_model

    def route(self, query: str, history: Optional[list] = None) -> RouteDecision:
        """
        路由决策主入口

        Args:
            query: 用户查询
            history: 对话历史（可选）

        Returns:
            RouteDecision: 路由决策
        """
        # 1. 意图分类
        decision = self.classifier.classify(query)

        # 2. 置信度检查
        if decision.confidence < self.confidence_threshold:
            logger.info(f"置信度 {decision.confidence:.2f} 低于阈值 {self.confidence_threshold}，降级为 hybrid")
            decision = RouteDecision(
                route="hybrid",
                confidence=decision.confidence,
                reasoning=f"原始分类置信度不足，降级为 hybrid。原始: {decision.reasoning}",
            )

        # 3. 对话历史增强（如果有）
        if history:
            decision = self._enhance_with_history(decision, history)

        logger.info(f"最终路由决策: {decision.route} (置信度: {decision.confidence:.2f})")
        return decision

    def _enhance_with_history(self, decision: RouteDecision, history: list) -> RouteDecision:
        """根据对话历史增强路由决策"""
        # 如果历史中有图相关查询，增强 graph 路由倾向
        graph_history = sum(
            1 for h in history[-3:]  # 只看最近3轮
            if any(kw in h.get("query", "").lower() for kw in ["关系", "关联", "比较", "对比"])
        )
        if graph_history >= 2 and decision.route == "standard":
            return RouteDecision(
                route="hybrid",
                confidence=decision.confidence,
                reasoning=f"历史对话中有 {graph_history} 轮关系查询，升级为 hybrid",
            )
        return decision

    def get_fallback_route(self, failed_route: RouteType) -> RouteType:
        """获取 Fallback 路由"""
        fallback_map = {
            "graph": "standard",
            "standard": "hybrid",
            "hybrid": "standard",
        }
        return fallback_map.get(failed_route, "standard")