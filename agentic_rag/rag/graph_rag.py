"""
GraphRAG 引擎模块
================
基于知识图谱的检索增强生成。
支持实体抽取、关系抽取、图构建、图遍历检索和社区摘要。
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger
import json
import re
import hashlib


@dataclass
class GraphRAGResult:
    """GraphRAG 结果"""
    answer: str
    sources: List[dict] = field(default_factory=list)
    entities: List[dict] = field(default_factory=list)
    relations: List[dict] = field(default_factory=list)
    communities: List[dict] = field(default_factory=list)
    route: str = "graph"
    confidence: float = 0.0
    latency_ms: float = 0.0


class EntityRelationExtractor:
    """实体关系抽取器——从文档中提取实体和关系"""

    EXTRACTION_PROMPT = """你是一个信息抽取专家。从以下文本中提取所有重要实体和它们之间的关系。

实体类型：person（人物）, organization（组织）, location（地点）, concept（概念）, event（事件）
关系类型：works_for（工作于）, located_in（位于）, part_of（属于）, 
          founded_by（创立者）, ceo_of（CEO）, acquired（收购）, 
          collaborated_with（合作）, related_to（相关）, produces（生产）

文本：
{text}

请以 JSON 格式输出：
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型", "description": "简要描述"}}
    ],
    "relations": [
        {{"source": "源实体名称", "target": "目标实体名称", "type": "关系类型", "description": "关系描述"}}
    ]
}}
"""

    def __init__(self, llm_client=None, llm_model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def extract(self, text: str) -> Tuple[List[dict], List[dict]]:
        """从文本中提取实体和关系"""
        if self.llm_client:
            return self._extract_with_llm(text)
        return self._extract_with_rules(text)

    def _extract_with_llm(self, text: str) -> Tuple[List[dict], List[dict]]:
        """使用 LLM 抽取"""
        try:
            prompt = self.EXTRACTION_PROMPT.format(text=text[:3000])
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            logger.info(f"实体抽取: {len(entities)} 实体, {len(relations)} 关系")
            return entities, relations
        except Exception as e:
            logger.warning(f"LLM 抽取失败: {e}，使用规则抽取")
            return self._extract_with_rules(text)

    def _extract_with_rules(self, text: str) -> Tuple[List[dict], List[dict]]:
        """基于规则的实体抽取（降级方案）"""
        import jieba.analyse

        entities = []
        relations = []

        # 使用 TF-IDF 提取关键词作为概念实体
        keywords = jieba.analyse.extract_tags(text, topK=20, withWeight=True)
        for kw, weight in keywords:
            if len(kw) >= 2:
                entities.append({
                    "name": kw,
                    "type": "concept",
                    "description": f"关键词，权重 {weight:.3f}",
                })

        # 简单的人名识别
        import jieba.posseg as pseg
        for word, flag in pseg.cut(text):
            if flag == "nr" and len(word) >= 2:
                if not any(e["name"] == word for e in entities):
                    entities.append({
                        "name": word,
                        "type": "person",
                        "description": "人物",
                    })

        return entities, relations


class GraphRAGEngine:
    """
    GraphRAG 引擎

    流程：
    1. 文档 → 实体抽取 → 关系抽取 → 图构建
    2. 查询 → 实体识别 → 图遍历 → 子图检索 → 答案生成
    """

    def __init__(
        self,
        graph_store=None,
        vector_store=None,
        embedder=None,
        reranker=None,
        llm_client=None,
        llm_model: str = "gpt-4o-mini",
        extractor=None,
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.extractor = extractor or EntityRelationExtractor(llm_client, llm_model)

    def build_graph_from_documents(self, documents: List[dict]) -> dict:
        """
        从文档构建知识图谱

        Args:
            documents: 文档列表，每项包含 id, content, metadata

        Returns:
            构建统计信息
        """
        stats = {"entities": 0, "relations": 0, "documents": len(documents)}

        for doc in documents:
            text = doc.get("content", "")
            if not text:
                continue

            # 抽取实体和关系
            entities, relations = self.extractor.extract(text)

            # 添加到图存储
            from agentic_rag.memory.graph_store import GraphEntity, GraphRelation

            doc_id = doc.get("id") or f"doc_{hashlib.md5(text.encode('utf-8')).hexdigest()[:12]}"
            for ent in entities:
                entity = GraphEntity(
                    id=f"ent_{hashlib.md5(ent['name'].encode('utf-8')).hexdigest()[:12]}",
                    name=ent["name"],
                    type=ent.get("type", "concept"),
                    properties={
                        "description": ent.get("description", ""),
                        "source_doc": doc_id,
                    },
                )
                if self.graph_store.add_entity(entity):
                    stats["entities"] += 1

            for rel in relations:
                # 查找或创建源/目标实体
                source_entities = self.graph_store.search_entity(rel["source"])
                target_entities = self.graph_store.search_entity(rel["target"])

                if source_entities and target_entities:
                    relation = GraphRelation(
                        source_id=source_entities[0].id,
                        target_id=target_entities[0].id,
                        relation_type=rel.get("type", "related_to"),
                        properties={"description": rel.get("description", "")},
                    )
                    if self.graph_store.add_relation(relation):
                        stats["relations"] += 1

        # 社区检测
        try:
            communities = self.graph_store.detect_communities()
            stats["communities"] = len(communities)
        except Exception as e:
            logger.warning(f"社区检测失败: {e}")
            stats["communities"] = 0

        logger.info(f"图构建完成: {stats}")
        return stats

    def query(
        self,
        query_text: str,
        top_k: int = 20,
        use_community: bool = True,
    ) -> GraphRAGResult:
        """
        执行 GraphRAG 查询

        Args:
            query_text: 用户查询
            top_k: 检索数量
            use_community: 是否使用社区摘要

        Returns:
            GraphRAGResult
        """
        import time
        start_time = time.time()

        # 1. 从查询中提取实体
        query_entities, _ = self.extractor.extract(query_text)
        query_entity_names = [e["name"] for e in query_entities]

        # 2. 在图存储中查找匹配实体（图存储未配置时仅依赖向量检索）
        matched_entities = []
        if self.graph_store:
            for name in query_entity_names:
                entities = self.graph_store.search_entity(name)
                matched_entities.extend(entities)
        else:
            logger.warning("图存储未配置，GraphRAG 仅使用向量检索补充")

        # 3. 图遍历——获取邻居和子图
        graph_context_parts = []
        all_relations = []

        for entity in matched_entities[:5]:  # 限制数量
            neighbors = self.graph_store.get_neighbors(entity.id, max_depth=2) if self.graph_store else []
            for neighbor, relation in neighbors:
                graph_context_parts.append(
                    f"{entity.name} --[{relation.relation_type}]--> {neighbor.name}"
                )
                all_relations.append({
                    "source": entity.name,
                    "target": neighbor.name,
                    "type": relation.relation_type,
                })

        # 4. 社区摘要（可选）——只检测一次，供上下文与响应复用
        community_context = ""
        communities = []
        if use_community and self.graph_store:
            try:
                communities = self.graph_store.detect_communities()
                for comm in communities[:3]:
                    summary = self.graph_store.get_community_summary(comm.community_id)
                    if summary:
                        community_context += f"\n{summary}"
            except Exception as e:
                logger.warning(f"社区摘要获取失败: {e}")

        # 5. 向量检索补充（混合检索）
        vector_context = ""
        if self.vector_store and self.embedder:
            try:
                query_vec = self.embedder.embed_query(query_text)
                vector_results = self.vector_store.search(query_vec, top_k=top_k)
                if vector_results:
                    vector_context = "\n".join([
                        r.payload.get("content", "")[:500] for r in vector_results[:5]
                    ])
            except Exception as e:
                logger.warning(f"向量检索补充失败: {e}")

        # 6. 构建最终上下文
        context_parts = []
        if graph_context_parts:
            context_parts.append("【知识图谱信息】\n" + "\n".join(graph_context_parts[:20]))
        if community_context:
            context_parts.append("【社区摘要】\n" + community_context)
        if vector_context:
            context_parts.append("【相关文档】\n" + vector_context)

        context = "\n\n".join(context_parts)

        # 7. 生成答案
        answer = self._generate_answer(query_text, context)

        elapsed = (time.time() - start_time) * 1000

        return GraphRAGResult(
            answer=answer,
            sources=[{"content": part[:200], "source": "graph"} for part in graph_context_parts[:5]],
            entities=[{"name": e.name, "type": e.type} for e in matched_entities],
            relations=all_relations,
            communities=[{"id": c.community_id, "size": len(c.entities)} for c in communities[:5]] if use_community else [],
            route="graph",
            confidence=min(0.9, len(matched_entities) * 0.15 + 0.1),
            latency_ms=elapsed,
        )

    def _generate_answer(self, query: str, context: str) -> str:
        """使用 LLM 生成答案"""
        if not self.llm_client:
            return self._fallback_generate(query, context)

        system = (
            "你是一个基于知识图谱的智能问答助手。请基于提供的图谱信息和文档，准确回答用户问题。\n"
            "要求：\n"
            "1. 优先使用图谱中的关系信息回答关系类问题\n"
            "2. 如果信息不足，明确说明\n"
            "3. 回答使用中文\n"
            "4. 可以引用实体之间的关系路径"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"参考信息：\n{context}\n\n用户问题：{query}"},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return self._fallback_generate(query, context)

    def _fallback_generate(self, query: str, context: str) -> str:
        if not context.strip():
            return "抱歉，知识库中没有找到相关信息。"
        return f"基于知识图谱检索到的信息：\n\n{context[:500]}..."