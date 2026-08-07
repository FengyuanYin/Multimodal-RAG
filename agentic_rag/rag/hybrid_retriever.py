"""
混合检索器模块
============
融合向量检索、关键词检索（BM25）和图检索结果。
使用 RRF（倒数排名融合）算法进行结果融合。
"""

from typing import List, Optional, Dict, Any
from loguru import logger
import math

from agentic_rag.processing.reranker import ScoredDocument


class BM25Retriever:
    """BM25 关键词检索器"""

    def __init__(self):
        self._bm25 = None
        self._documents: List[ScoredDocument] = []
        self._corpus: List[str] = []

    def index(self, documents: List[ScoredDocument], append: bool = False):
        """构建 BM25 索引

        Args:
            documents: 待索引文档
            append: 是否追加到已有索引（多次摄入时使用）
        """
        from rank_bm25 import BM25Okapi
        import jieba

        if append and self._bm25 is not None:
            self._documents.extend(documents)
        else:
            self._documents = documents

        self._corpus = []
        for doc in self._documents:
            # 中文分词（统一小写，保证大小写不敏感匹配）
            words = [w.lower() for w in jieba.cut(doc.content)]
            self._corpus.append(words)

        self._bm25 = BM25Okapi(self._corpus)
        logger.info(f"BM25 索引构建完成: {len(self._documents)} 篇文档")

    def search(self, query: str, top_k: int = 10) -> List[ScoredDocument]:
        """BM25 检索"""
        if not self._bm25:
            logger.warning("BM25 索引为空")
            return []

        import jieba
        query_words = [w.lower() for w in jieba.cut(query)]
        scores = self._bm25.get_scores(query_words)

        scored = []
        for i, score in enumerate(scores):
            # 注意：rank_bm25 在单文档语料 / 词出现在全部文档时 IDF 为负，
            # 命中词的分数可能 ≤ 0，不能按 score > 0 过滤；0 才表示完全未命中
            if score != 0:
                doc = self._documents[i]
                scored.append(ScoredDocument(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=float(score),
                    metadata=doc.metadata,
                    modality=doc.modality,
                    source="keyword",
                ))

        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]


class HybridRetriever:
    """
    混合检索器
    融合向量检索、关键词检索和图检索结果。
    使用 RRF（倒数排名融合）算法。
    """

    def __init__(
        self,
        vector_store=None,
        graph_store=None,
        embedder=None,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.bm25 = BM25Retriever()
        self._index_built = False

    def build_index(self, documents: List[ScoredDocument], append: bool = False):
        """构建检索索引

        Args:
            documents: 待索引文档
            append: 是否追加到已有索引（多次摄入时使用）
        """
        self.bm25.index(documents, append=append)
        self._index_built = True

    def load_persisted_index(self, path: Optional[str] = None) -> int:
        """从持久化文件重建 BM25 索引（服务重启 / 新进程时调用）

        Args:
            path: JSONL 分块文件路径，默认 data/index/chunks.jsonl

        Returns:
            加载的分块数量
        """
        import json
        import os

        path = path or os.path.join(os.getcwd(), "data", "index", "chunks.jsonl")
        if not os.path.exists(path):
            return 0

        chunks = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    chunks.append(ScoredDocument(
                        doc_id=data.get("doc_id", ""),
                        content=data.get("content", ""),
                        score=1.0,
                        metadata=data.get("metadata", {}),
                        modality=data.get("modality", "text"),
                    ))
        except Exception as e:
            logger.warning(f"加载持久化索引失败: {e}")
            return 0

        if chunks:
            self.build_index(chunks)  # 全量重建（文件为全量快照）
            logger.info(f"从持久化文件重建 BM25 索引: {len(chunks)} 个分块")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        use_vector: bool = True,
        use_keyword: bool = True,
        use_graph: bool = False,
        query_embedding: Optional[List[float]] = None,
    ) -> List[ScoredDocument]:
        """
        混合检索主入口

        Args:
            query: 查询文本
            top_k: 最终返回数量
            use_vector: 是否使用向量检索
            use_keyword: 是否使用关键词检索
            use_graph: 是否使用图检索
            query_embedding: 预计算的查询向量

        Returns:
            融合后的文档列表
        """
        all_results = []

        # 1. 向量检索
        if use_vector and self.vector_store and self.embedder:
            try:
                if not query_embedding:
                    query_embedding = self.embedder.embed_query(query)
                vector_results = self.vector_store.search(query_embedding, top_k=top_k)
                for r in vector_results:
                    all_results.append(ScoredDocument(
                        doc_id=r.id,
                        content=r.content or r.payload.get("content", ""),
                        score=r.score,
                        metadata=r.payload,
                        source="vector",
                    ))
                logger.debug(f"向量检索: {len(vector_results)} 条结果")
            except Exception as e:
                logger.warning(f"向量检索失败: {e}")

        # 2. 关键词检索
        if use_keyword and self._index_built:
            try:
                keyword_results = self.bm25.search(query, top_k=top_k)
                all_results.extend(keyword_results)
                logger.debug(f"关键词检索: {len(keyword_results)} 条结果")
            except Exception as e:
                logger.warning(f"关键词检索失败: {e}")

        # 3. 图检索
        if use_graph and self.graph_store:
            try:
                graph_results = self._graph_retrieve(query, top_k)
                all_results.extend(graph_results)
                logger.debug(f"图检索: {len(graph_results)} 条结果")
            except Exception as e:
                logger.warning(f"图检索失败: {e}")

        # 4. 结果融合（RRF）
        fused = self._reciprocal_rank_fusion(all_results, top_k)
        logger.info(f"混合检索完成: 原始 {len(all_results)} 条, 融合后 {len(fused)} 条")
        return fused

    def _graph_retrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        """图检索：从查询中提取实体，遍历图获取相关文档"""
        results = []

        # 简单实体提取（后续可集成 NER）
        import jieba.analyse
        keywords = jieba.analyse.extract_tags(query, topK=5)

        for kw in keywords:
            entities = self.graph_store.search_entity(kw)
            for entity in entities:
                neighbors = self.graph_store.get_neighbors(entity.id, max_depth=2)
                for neighbor, relation in neighbors:
                    # 将邻居实体信息作为文档返回
                    content = f"{entity.name} --[{relation.relation_type}]--> {neighbor.name}"
                    results.append(ScoredDocument(
                        doc_id=f"graph_{entity.id}_{neighbor.id}",
                        content=content,
                        score=relation.weight,
                        metadata={
                            "source_entity": entity.name,
                            "target_entity": neighbor.name,
                            "relation": relation.relation_type,
                        },
                        modality="text",
                        source="graph",
                    ))

        return results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        documents: List[ScoredDocument],
        top_k: int,
        k: int = 60,
    ) -> List[ScoredDocument]:
        """
        RRF（倒数排名融合）算法

        Args:
            documents: 多源文档列表
            top_k: 返回数量
            k: RRF 常数（通常 60）

        Returns:
            融合排序后的文档列表
        """
        if not documents:
            return []

        # 按来源分组并排序
        from collections import defaultdict
        sources = defaultdict(list)
        for doc in documents:
            sources[doc.source].append(doc)

        for source, docs in sources.items():
            docs.sort(key=lambda d: d.score, reverse=True)

        # 计算 RRF 分数
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, ScoredDocument] = {}

        for source, docs in sources.items():
            for rank, doc in enumerate(docs):
                rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank + 1)
                if doc.doc_id not in doc_map:
                    doc_map[doc.doc_id] = doc

        # 按 RRF 分数排序
        ranked = sorted(
            [(doc_id, score) for doc_id, score in rrf_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for doc_id, score in ranked[:top_k]:
            doc = doc_map[doc_id]
            doc.score = score
            results.append(doc)

        return results

    def retrieve_with_rewrite(
        self,
        query: str,
        rewritten_query: 'RewrittenQuery',
        top_k: int = 20,
    ) -> List[ScoredDocument]:
        """
        使用重写后的查询进行混合检索

        Args:
            query: 原始查询
            rewritten_query: 重写后的查询对象
            top_k: 返回数量

        Returns:
            融合后的文档列表
        """
        all_results = []

        # 1. 用原始查询检索
        original_results = self.retrieve(query, top_k=top_k)
        all_results.extend(original_results)

        # 2. 用改写版本检索
        for variant in rewritten_query.variants:
            variant_results = self.retrieve(variant, top_k=top_k // 2)
            all_results.extend(variant_results)

        # 3. 用子问题检索
        for sub_query in rewritten_query.sub_queries:
            sub_results = self.retrieve(sub_query, top_k=top_k // 2)
            all_results.extend(sub_results)

        # 4. 用 HyDE 文档检索（如果有）
        if rewritten_query.hyde_answer and self.embedder:
            hyde_embedding = self.embedder.embed_query(rewritten_query.hyde_answer)
            hyde_results = self.retrieve(
                query, top_k=top_k // 2, query_embedding=hyde_embedding
            )
            all_results.extend(hyde_results)

        # 5. 最终融合
        fused = self._reciprocal_rank_fusion(all_results, top_k)
        return fused