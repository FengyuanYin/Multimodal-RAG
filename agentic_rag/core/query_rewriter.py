"""
查询重写器模块
============
对用户原始问题进行改写，提升检索质量。
支持查询扩展、分解、澄清、HyDE 等多种策略。
"""

from typing import List, Optional
from dataclasses import dataclass, field
from loguru import logger
import json
import re


@dataclass
class RewrittenQuery:
    """查询重写结果"""
    original: str
    variants: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    strategy: str = "none"
    hyde_answer: Optional[str] = None


class QueryRewriter:
    """
    查询重写器
    支持多种重写策略，可根据查询类型自动选择策略组合。
    """

    # 查询扩展提示
    EXPANSION_PROMPT = """你是一个查询扩展专家。给定用户查询，生成 3 个不同表述的改写版本，
帮助检索系统找到更相关的信息。

要求：
- 保持原意不变
- 使用不同的表达方式和同义词
- 每个版本独立成行
- 不要包含序号或额外说明

原始查询：{query}

改写版本："""

    # 查询分解提示
    DECOMPOSITION_PROMPT = """你是一个查询分解专家。将复杂查询分解为多个简单的子问题，
每个子问题应该独立可检索。

原始查询：{query}

请以 JSON 数组格式输出子问题列表：
["子问题1", "子问题2", ...]"""

    # HyDE 提示
    HYDE_PROMPT = """给定一个查询，生成一段假设性的回答文档。
这段文档应该包含可能出现在真实相关文档中的内容和风格。
不要直接回答问题，而是生成一段看起来像是从知识库中摘录的文字。

查询：{query}

假设性文档："""

    def __init__(self, llm_client=None, llm_model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def rewrite(self, query: str, strategy: str = "auto") -> RewrittenQuery:
        """
        重写查询

        Args:
            query: 原始查询
            strategy: 重写策略 (auto | expansion | decomposition | hyde | all)

        Returns:
            RewrittenQuery: 重写结果
        """
        if strategy == "auto":
            strategy = self._select_strategy(query)

        result = RewrittenQuery(original=query, strategy=strategy)

        if strategy in ("expansion", "all"):
            result.variants = self._expand_query(query)

        if strategy in ("decomposition", "all"):
            result.sub_queries = self._decompose_query(query)

        if strategy in ("hyde", "all"):
            result.hyde_answer = self._generate_hyde(query)

        logger.info(f"查询重写完成: strategy={strategy}, variants={len(result.variants)}, "
                    f"sub_queries={len(result.sub_queries)}")
        return result

    def _select_strategy(self, query: str) -> str:
        """自动选择重写策略"""
        query_lower = query.lower()

        # 复杂问题 -> 分解
        if any(kw in query_lower for kw in ["并且", "同时", "分别", "各自", "and", "both"]):
            return "decomposition"

        # 模糊/简短查询 -> 扩展
        if len(query) < 15 or any(kw in query_lower for kw in ["它", "这个", "那个", "it", "this", "that"]):
            return "expansion"

        # 需要推理的查询 -> HyDE
        if any(kw in query_lower for kw in ["为什么", "如何", "怎样", "why", "how"]):
            return "hyde"

        # 默认：扩展 + 分解
        return "all"

    def _expand_query(self, query: str) -> List[str]:
        """查询扩展"""
        if not self.llm_client:
            return self._rule_based_expand(query)

        try:
            prompt = self.EXPANSION_PROMPT.format(query=query)
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            variants = [v.strip() for v in content.split("\n") if v.strip()]
            # 去重并限制数量
            seen = set()
            unique = []
            for v in variants:
                if v not in seen and v != query:
                    seen.add(v)
                    unique.append(v)
            return unique[:5]
        except Exception as e:
            logger.warning(f"LLM 查询扩展失败: {e}，使用规则扩展")
            return self._rule_based_expand(query)

    def _rule_based_expand(self, query: str) -> List[str]:
        """基于规则的查询扩展（降级方案）"""
        variants = [query]

        # 添加同义表述
        if "是" in query:
            variants.append(query.replace("是", "是指"))
            variants.append(query.replace("是", "的定义"))

        if "如何" in query:
            variants.append(query.replace("如何", "怎样"))
            variants.append(query.replace("如何", "怎么"))

        if "为什么" in query:
            variants.append(query.replace("为什么", "为何"))
            variants.append(query.replace("为什么", "原因是什么"))

        return list(set(variants))[:5]

    def _decompose_query(self, query: str) -> List[str]:
        """查询分解"""
        if not self.llm_client:
            return self._rule_based_decompose(query)

        try:
            prompt = self.DECOMPOSITION_PROMPT.format(query=query)
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            content = response.choices[0].message.content
            # 尝试解析 JSON
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    for key in ["sub_queries", "questions", "subquestions"]:
                        if key in data and isinstance(data[key], list):
                            return data[key]
            except json.JSONDecodeError:
                # 尝试按行解析
                lines = [l.strip().strip('"').strip("- ") for l in content.split("\n") if l.strip()]
                return [l for l in lines if l and not l.startswith("{") and not l.startswith("}")]
        except Exception as e:
            logger.warning(f"LLM 查询分解失败: {e}，使用规则分解")
            return self._rule_based_decompose(query)

    def _rule_based_decompose(self, query: str) -> List[str]:
        """基于规则的查询分解"""
        sub_queries = [query]

        # 按 "和"、"与"、"、" 分割
        parts = re.split(r" 和 | 与 |、|, ", query)
        if len(parts) > 1:
            sub_queries.extend(parts)

        return list(set(sub_queries))[:5]

    def _generate_hyde(self, query: str) -> Optional[str]:
        """生成假设性文档（HyDE）"""
        if not self.llm_client:
            return None

        try:
            prompt = self.HYDE_PROMPT.format(query=query)
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"HyDE 生成失败: {e}")
            return None