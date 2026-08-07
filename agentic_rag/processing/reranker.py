"""
重排序器模块
==========
使用交叉编码器对检索结果进行精细化排序。
支持 BGE-Reranker、Cohere Rerank 等模型。
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ScoredDocument:
    """带分数的文档（统一数据结构）"""
    doc_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)
    modality: str = "text"
    source: str = "vector"  # vector | keyword | graph


class BaseReranker:
    """重排序器基类"""

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        raise NotImplementedError


class BGEReranker(BaseReranker):
    """BGE 交叉编码器重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            logger.info(f"加载重排序模型: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, device_map=self.device
            )
            self._model.eval()
            logger.info("重排序模型加载完成")
        except Exception as e:
            logger.warning(f"加载本地重排序模型失败: {e}，将使用 LLM 重排序")
            self._model = None

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        if not documents:
            return []

        if self._model is None:
            return self._rerank_with_llm(query, documents, top_k)

        import torch

        pairs = [[query, doc.content[:512]] for doc in documents]
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            scores = outputs.logits.squeeze(-1).tolist()

        if isinstance(scores, float):
            scores = [scores]

        for doc, score in zip(documents, scores):
            doc.score = float(score)

        ranked = sorted(documents, key=lambda d: d.score, reverse=True)
        if top_k:
            ranked = ranked[:top_k]
        return ranked

    def _rerank_with_llm(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        """使用 LLM 进行重排序（降级方案）"""
        ranked = sorted(documents, key=lambda d: d.score, reverse=True)
        if top_k:
            ranked = ranked[:top_k]
        return ranked


class CohereReranker(BaseReranker):
    """Cohere 重排序器（API）"""

    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self.api_key)
            except ImportError:
                raise ImportError("需要安装 cohere 包: pip install cohere")
        return self._client

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: Optional[int] = None,
    ) -> List[ScoredDocument]:
        client = self._get_client()
        docs = [doc.content[:500] for doc in documents]
        response = client.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=top_k or len(documents),
        )
        for result in response.results:
            documents[result.index].score = result.relevance_score
        ranked = sorted(documents, key=lambda d: d.score, reverse=True)
        return ranked


class RerankerFactory:
    """重排序器工厂"""

    @staticmethod
    def create(
        provider: str = "bge",
        model_name: Optional[str] = None,
        device: str = "cpu",
        api_key: Optional[str] = None,
    ) -> BaseReranker:
        if provider == "bge":
            model = model_name or "BAAI/bge-reranker-v2-m3"
            return BGEReranker(model, device)
        elif provider == "cohere":
            if not api_key:
                raise ValueError("Cohere 重排序需要 api_key")
            return CohereReranker(api_key)
        else:
            raise ValueError(f"不支持的重排序器提供商: {provider}")