"""
Agentic GraphRAG 系统
====================
基于 GraphRAG + Agentic RAG 的智能问答系统：
- 混合路由：根据用户提问自适应选择 Standard RAG / GraphRAG / Hybrid
- 多模态记忆：文本 / 图片 / 表格 / PDF 统一解析与检索
- 高级 RAG 模式：查询重写、混合检索（向量+BM25+图遍历）、重排序

快速开始：
    from agentic_rag import AgenticRAG

    rag = AgenticRAG()
    rag.ingest_text("人工智能公司深度智能专注于大语言模型研发。")
    answer = rag.query("深度智能公司是做什么的？")
"""

from agentic_rag.client import AgenticRAG

# 核心数据模型
from agentic_rag.core.orchestrator import AgenticOrchestrator, QueryRequest, QueryResponse

# 核心组件
from agentic_rag.core.hybrid_router import HybridRouter, RouteDecision, IntentClassifier
from agentic_rag.core.query_rewriter import QueryRewriter, RewrittenQuery

# RAG 引擎
from agentic_rag.rag.standard_rag import StandardRAGEngine, RAGResult
from agentic_rag.rag.graph_rag import GraphRAGEngine, GraphRAGResult, EntityRelationExtractor
from agentic_rag.rag.hybrid_retriever import HybridRetriever, BM25Retriever

# 存储与解析
from agentic_rag.memory.vector_store import VectorStoreFactory
from agentic_rag.memory.graph_store import GraphStoreFactory
from agentic_rag.memory.multi_modal_parser import MultiModalParser

# 处理组件
from agentic_rag.processing.embedders import EmbedderFactory
from agentic_rag.processing.reranker import RerankerFactory, ScoredDocument
from agentic_rag.processing.chunker import TextChunker, SemanticChunker, get_chunker

# 配置
from agentic_rag.config import Settings, settings

# 工厂与服务
from agentic_rag.factory import build_orchestrator
from agentic_rag.service import ingest_documents

__version__ = "0.1.0"

__all__ = [
    # 高层客户端
    "AgenticRAG",
    # 编排器
    "AgenticOrchestrator",
    "QueryRequest",
    "QueryResponse",
    # 混合路由
    "HybridRouter",
    "RouteDecision",
    "IntentClassifier",
    # 查询重写
    "QueryRewriter",
    "RewrittenQuery",
    # RAG 引擎
    "StandardRAGEngine",
    "RAGResult",
    "GraphRAGEngine",
    "GraphRAGResult",
    "EntityRelationExtractor",
    "HybridRetriever",
    "BM25Retriever",
    # 存储与解析
    "VectorStoreFactory",
    "GraphStoreFactory",
    "MultiModalParser",
    # 处理组件
    "EmbedderFactory",
    "RerankerFactory",
    "ScoredDocument",
    "TextChunker",
    "SemanticChunker",
    "get_chunker",
    # 配置 / 工厂 / 服务
    "Settings",
    "settings",
    "build_orchestrator",
    "ingest_documents",
    "__version__",
]
