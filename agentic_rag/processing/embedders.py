"""
嵌入器模块
=========
支持多种嵌入模型：OpenAI、BGE-M3、Sentence-Transformers。
提供统一的嵌入接口。
"""

from typing import List, Optional, Union
import numpy as np
from loguru import logger


class BaseEmbedder:
    """嵌入器基类"""

    def __init__(self, model_name: str, device: str = "cpu", dim: int = 1024):
        self.model_name = model_name
        self.device = device
        self.dim = dim
        self._model = None

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """统一嵌入接口"""
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询（部分模型使用不同的指令前缀）"""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self.dim


class BGEEmbedder(BaseEmbedder):
    """BGE-M3 嵌入器（本地）"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", dim: int = 1024):
        super().__init__(model_name, device, dim)
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载嵌入模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"嵌入模型加载完成，维度: {self.dim}")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            raise

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        # BGE 使用指令前缀
        return self.embed([f"为这个句子生成表示以用于检索相关文章：{text}"])[0]


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI 嵌入器"""

    def __init__(self, model_name: str = "text-embedding-3-small", dim: int = 512, api_key: Optional[str] = None):
        super().__init__(model_name, "cpu", dim)
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=self.dim,
        )
        return [item.embedding for item in response.data]


class EmbedderFactory:
    """嵌入器工厂"""

    @staticmethod
    def create(
        provider: str = "bge",
        model_name: Optional[str] = None,
        device: str = "cpu",
        dim: int = 1024,
        api_key: Optional[str] = None,
    ) -> BaseEmbedder:
        if provider == "openai":
            model = model_name or "text-embedding-3-small"
            return OpenAIEmbedder(model, dim, api_key)
        elif provider == "bge":
            model = model_name or "BAAI/bge-m3"
            return BGEEmbedder(model, device, dim)
        else:
            raise ValueError(f"不支持的嵌入器提供商: {provider}")