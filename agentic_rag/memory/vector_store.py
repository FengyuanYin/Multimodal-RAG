"""
向量存储模块
===========
支持 ChromaDB 和 Qdrant 两种后端，提供统一的向量检索接口。
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import numpy as np
from loguru import logger


@dataclass
class VectorRecord:
    """向量存储记录"""
    id: str
    vector: List[float]
    payload: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """检索结果"""
    id: str
    score: float
    payload: dict = field(default_factory=dict)
    content: str = ""


class BaseVectorStore:
    """向量存储基类"""

    def __init__(self, collection_name: str = "default", embedding_dim: int = 1024):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

    def add(self, records: List[VectorRecord]) -> int:
        raise NotImplementedError

    def search(self, query_vector: List[float], top_k: int = 10, filter: Optional[dict] = None) -> List[SearchResult]:
        raise NotImplementedError

    def delete(self, ids: List[str]) -> bool:
        raise NotImplementedError

    def delete_collection(self, name: str) -> bool:
        """删除指定集合。删除当前使用的集合后会自动重建空集合以保持可用。"""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def list_collections(self) -> List[str]:
        raise NotImplementedError


class ChromaStore(BaseVectorStore):
    """ChromaDB 向量存储"""

    def __init__(self, collection_name: str = "default", embedding_dim: int = 1024, persist_dir: str = "./data/vector_db"):
        super().__init__(collection_name, embedding_dim)
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self):
        try:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB 初始化完成: collection={self.collection_name}, dim={self.embedding_dim}")
        except Exception as e:
            logger.error(f"ChromaDB 初始化失败: {e}")
            raise

    def add(self, records: List[VectorRecord]) -> int:
        if not records:
            return 0
        ids = [r.id for r in records]
        vectors = [r.vector for r in records]
        metadatas = [r.payload for r in records]

        # 确保向量维度匹配
        for v in vectors:
            if len(v) != self.embedding_dim:
                logger.warning(f"向量维度不匹配: 期望 {self.embedding_dim}, 实际 {len(v)}")

        self._collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
        )
        return len(ids)

    def search(self, query_vector: List[float], top_k: int = 10, filter: Optional[dict] = None) -> List[SearchResult]:
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, 100),
            where=filter,
        )
        if not results["ids"]:
            return []

        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append(SearchResult(
                id=results["ids"][0][i],
                score=1.0 - results["distances"][0][i] if results["distances"] else 0.0,
                payload=results["metadatas"][0][i] if results["metadatas"] else {},
                content=results["metadatas"][0][i].get("content", "") if results["metadatas"] else "",
            ))
        return search_results

    def delete(self, ids: List[str]) -> bool:
        try:
            self._collection.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"删除失败: {e}")
            return False

    def delete_collection(self, name: str) -> bool:
        try:
            self._client.delete_collection(name)
            if name == self.collection_name:
                # 删除的是当前使用的集合，重建空集合以保持服务可用
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            logger.info(f"ChromaDB 集合已删除: {name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    def count(self) -> int:
        return self._collection.count()

    def list_collections(self) -> List[str]:
        return [c.name for c in self._client.list_collections()]


class QdrantStore(BaseVectorStore):
    """Qdrant 向量存储"""

    def __init__(self, collection_name: str = "default", embedding_dim: int = 1024,
                 host: str = "localhost", port: int = 6333):
        super().__init__(collection_name, embedding_dim)
        self.host = host
        self.port = port
        self._client = None
        self._init_client()

    @staticmethod
    def _stable_point_id(id: str) -> int:
        """生成稳定的非负 64 位整数 ID（避免 Python hash() 跨进程随机化）"""
        return int(hashlib.sha256(id.encode("utf-8")).hexdigest()[:16], 16) % (2**63)

    def _init_client(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import VectorParams, Distance
            self._client = QdrantClient(host=self.host, port=self.port)
            # 检查集合是否存在，不存在则创建
            collections = self._client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
                )
            logger.info(f"Qdrant 初始化完成: collection={self.collection_name}")
        except Exception as e:
            logger.error(f"Qdrant 初始化失败: {e}")
            raise

    def add(self, records: List[VectorRecord]) -> int:
        from qdrant_client.http.models import PointStruct
        points = [
            PointStruct(
                id=self._stable_point_id(r.id),
                vector=r.vector,
                payload=r.payload,
            )
            for r in records
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, query_vector: List[float], top_k: int = 10, filter: Optional[dict] = None) -> List[SearchResult]:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        qdrant_filter = None
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload,
                content=r.payload.get("content", ""),
            )
            for r in results
        ]

    def delete(self, ids: List[str]) -> bool:
        try:
            point_ids = [self._stable_point_id(id) for id in ids]
            self._client.delete(collection_name=self.collection_name, points_selector=point_ids)
            return True
        except Exception as e:
            logger.error(f"删除失败: {e}")
            return False

    def delete_collection(self, name: str) -> bool:
        try:
            self._client.delete_collection(collection_name=name)
            if name == self.collection_name:
                # 删除的是当前使用的集合，重建空集合以保持服务可用
                self._init_client()
            logger.info(f"Qdrant 集合已删除: {name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    def count(self) -> int:
        return self._client.count(collection_name=self.collection_name).count

    def list_collections(self) -> List[str]:
        return [c.name for c in self._client.get_collections().collections]


class VectorStoreFactory:
    """向量存储工厂"""

    @staticmethod
    def create(
        db_type: str = "chroma",
        collection_name: str = "default",
        embedding_dim: int = 1024,
        persist_dir: str = "./data/vector_db",
        host: str = "localhost",
        port: int = 6333,
    ) -> BaseVectorStore:
        if db_type == "qdrant":
            return QdrantStore(collection_name, embedding_dim, host, port)
        return ChromaStore(collection_name, embedding_dim, persist_dir)