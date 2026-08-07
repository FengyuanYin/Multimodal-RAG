"""
系统配置模块
===========
使用 pydantic-settings 管理所有配置项，支持环境变量覆盖。
"""

from pydantic_settings import BaseSettings
from typing import Optional
from loguru import logger


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

    # ── VLM（视觉语言模型，多模态检索/图问答） ──
    vlm_provider: str = "openai"  # openai | litellm
    vlm_model: str = "gpt-4o-mini"
    vlm_api_key: Optional[str] = None
    vlm_base_url: Optional[str] = None

    # ── 多模态检索 ──
    # 基于知识图谱的"文本块 → 图片/表格"引用检索（RAG-Anything 风格）
    enable_multimodal_retrieval: bool = True
    # 媒体资产注册表持久化路径（图片 base64 / 表格文本）
    media_store_path: str = "./data/media/media_registry.json"
    # 媒体 base64 数据内存上限（MB），超出部分从磁盘懒加载
    media_max_memory_mb: int = 512
    # 文档、分块、媒体和引用的事务事实源；旧 JSON 注册表仅作兼容迁移。
    knowledge_db_path: str = "./data/knowledge/knowledge.db"
    request_timeout_seconds: float = 45.0
    max_upload_mb: int = 50
    allowed_cors_origins: str = "https://fengyuanyin.github.io,http://localhost:8000,http://127.0.0.1:8000"

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

    # ── VLM 配置状态（供 API/客户端判断） ──

    @property
    def vlm_configured(self) -> bool:
        """是否已配置可用的 VLM（模型名 + API Key）"""
        return bool(self.vlm_model and self.vlm_api_key)

    def is_vlm_configured(self) -> bool:
        return self.vlm_configured

    def vlm_config_dict(self, include_secret: bool = False) -> dict:
        """VLM 配置（脱敏）"""
        d = {
            "provider": self.vlm_provider,
            "model": self.vlm_model,
            "base_url": self.vlm_base_url or "",
            "configured": self.vlm_configured,
        }
        if include_secret:
            d["api_key"] = self.vlm_api_key or ""
        return d

    def save_vlm_config(self, provider: Optional[str] = None, model: Optional[str] = None,
                        api_key: Optional[str] = None, base_url: Optional[str] = None,
                        env_path: str = ".env") -> dict:
        """保存 VLM 配置：更新内存对象并写入 .env（重启后仍生效）"""
        if provider is not None:
            self.vlm_provider = provider or "openai"
        if model is not None:
            self.vlm_model = (model or "").strip()
        if api_key is not None:
            self.vlm_api_key = (api_key or "").strip() or None
        if base_url is not None:
            self.vlm_base_url = (base_url or "").strip() or None

        # 写入 .env（追加/更新 AGR_VLM_* 行）
        import os
        lines = []
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        updates = {
            "AGR_VLM_PROVIDER": self.vlm_provider,
            "AGR_VLM_MODEL": self.vlm_model,
            "AGR_VLM_API_KEY": self.vlm_api_key or "",
            "AGR_VLM_BASE_URL": self.vlm_base_url or "",
        }
        kept = [ln for ln in lines if not any(ln.strip().startswith(k + "=") for k in updates)]
        kept.extend(f"{k}={v}" for k, v in updates.items())
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept) + "\n")
        logger.info("VLM 配置已保存到 .env")
        return self.vlm_config_dict(include_secret=False)


settings = Settings()
