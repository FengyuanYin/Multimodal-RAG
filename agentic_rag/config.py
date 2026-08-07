"""
系统配置模块
===========
使用 pydantic-settings 管理所有配置项，支持环境变量覆盖。
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── 应用基础 ──
    app_name: str = "Agentic GraphRAG"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── API ──
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: Optional[str] = None  # 为空则不启用认证

    # ── 向量数据库 ──
    vector_db_type: str = "chroma"  # chroma | qdrant
    vector_db_path: str = "./data/vector_db"
    # Qdrant 配置
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ── 图数据库 ──
    graph_db_type: str = "networkx"  # networkx | neo4j
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None

    # ── 嵌入模型 ──
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"  # cpu | cuda

    # ── 重排序模型 ──
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"

    # ── LLM ──
    llm_provider: str = "openai"  # openai | litellm
    llm_model: str = "gpt-4o"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # ── 检索参数 ──
    top_k_initial: int = 20
    top_k_rerank: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 128

    # ── 路由 ──
    router_confidence_threshold: float = 0.6
    enable_fallback: bool = True

    # ── 日志 ──
    log_level: str = "INFO"
    log_file: Optional[str] = "./data/logs/app.log"

    model_config = {"env_prefix": "AGR_", "env_file": ".env", "extra": "ignore"}


settings = Settings()