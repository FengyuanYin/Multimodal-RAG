"""
AgenticRAG 高层客户端
====================
面向最终用户的 Facade 封装：懒加载初始化，一行代码接入完整 Agentic RAG 能力。

用法示例：
    from agentic_rag import AgenticRAG

    rag = AgenticRAG()
    rag.ingest_text("人工智能公司深度智能专注于大语言模型研发。")
    answer = rag.query("深度智能公司是做什么的？")
    print(answer)
"""

from typing import List, Dict, Any, Optional


class AgenticRAG:
    """Agentic GraphRAG 高层客户端"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Args:
            config: 配置覆盖字典（键名与 agentic_rag.config.Settings 字段一致，
                    如 {"llm_api_key": "...", "vector_db_path": "./data/vec"}）
            **kwargs: 便捷写法，等价于传入 config（如 llm_api_key="..."）
        """
        self._config = dict(config or {})
        self._config.update(kwargs)
        self._orchestrator = None

    # ── 初始化 ──

    def _ensure_orchestrator(self):
        """懒加载初始化编排器"""
        if self._orchestrator is None:
            from agentic_rag.factory import build_orchestrator
            from agentic_rag.config import settings

            if self._config:
                cfg = settings.model_copy(update=self._config)
            else:
                cfg = settings
            self._orchestrator = build_orchestrator(cfg)
        return self._orchestrator

    @property
    def orchestrator(self):
        """获取底层编排器实例（首次访问触发初始化）"""
        return self._ensure_orchestrator()

    # ── 文档摄入 ──

    def ingest(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        build_graph: bool = True,
    ) -> Dict[str, Any]:
        """
        摄入多模态文档

        Args:
            documents: 文档列表，每项形如
                {"content": "...", "modality": "text|image|table|pdf", "metadata": {...}}
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            build_graph: 是否构建知识图谱

        Returns:
            {"status", "doc_count", "chunk_count", "graph_stats", "message"}
        """
        from agentic_rag.service import ingest_documents
        return ingest_documents(
            self._ensure_orchestrator(),
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            build_graph=build_graph,
        )

    def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        build_graph: bool = True,
    ) -> Dict[str, Any]:
        """快捷摄入一段文本"""
        return self.ingest(
            [{"content": text, "modality": "text", "metadata": metadata or {}}],
            build_graph=build_graph,
        )

    # ── 问答 ──

    def query(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 5,
        rerank: bool = True,
        conversation_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        enable_multimodal: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        执行问答（自动路由）

        Args:
            query: 用户问题
            mode: 路由模式 auto|standard|graph|hybrid
            top_k: 返回结果数量
            rerank: 是否重排序
            conversation_id: 多轮对话 ID
            history: 对话历史
            enable_multimodal: 是否启用多模态检索（图片/表格引用，None=使用配置）

        Returns:
            dict: {"answer", "route", "confidence", "sources", "conversation_id", "latency_ms", "metadata"}
        """
        from agentic_rag.core.orchestrator import QueryRequest

        request = QueryRequest(
            query=query,
            conversation_id=conversation_id,
            history=history,
            mode=mode,
            top_k=top_k,
            rerank=rerank,
            enable_multimodal=enable_multimodal,
        )
        result = self._ensure_orchestrator().query(request)

        return {
            "answer": result.answer,
            "route": result.route,
            "confidence": result.confidence,
            "sources": result.sources,
            "conversation_id": result.conversation_id,
            "latency_ms": result.latency_ms,
            "metadata": result.metadata,
        }

    # ── 多模态 / VLM ──

    @property
    def vlm_configured(self) -> bool:
        """当前是否已配置 VLM 模型"""
        orch = self._orchestrator
        if orch is not None:
            return bool(getattr(orch, "vlm_client", None))
        from agentic_rag.config import settings
        return settings.vlm_configured

    def save_vlm_config(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
    ) -> dict:
        """
        保存 VLM 配置（写入 .env 并热更新当前实例）

        Args:
            model: VLM 模型名，如 gpt-4o / qwen-vl-max
            api_key: API Key（传 None 表示保持原值）
            base_url: Base URL（OpenAI 兼容）
            provider: 提供商

        Returns:
            dict: 保存后的配置（脱敏）
        """
        from agentic_rag.config import settings
        if self._config:
            cfg = settings.model_copy(update=self._config)
        else:
            cfg = settings
        result = cfg.save_vlm_config(
            provider=provider, model=model, api_key=api_key, base_url=base_url
        )
        # 热更新当前实例
        if self._orchestrator is not None:
            try:
                vlm_client = None
                if cfg.vlm_api_key:
                    from openai import OpenAI
                    vlm_client = OpenAI(
                        api_key=cfg.vlm_api_key,
                        base_url=cfg.vlm_base_url or cfg.llm_base_url,
                    )
                elif cfg.llm_api_key:
                    vlm_client = self._orchestrator.llm_client
                self._orchestrator.vlm_client = vlm_client
                self._orchestrator.vlm_model = cfg.vlm_model
                if self._orchestrator.standard_rag:
                    self._orchestrator.standard_rag.vlm_client = vlm_client
                    self._orchestrator.standard_rag.vlm_model = cfg.vlm_model
            except Exception as e:
                from loguru import logger
                logger.warning(f"VLM 客户端热更新失败（重启后生效）: {e}")
        return result

    # ── 系统状态 ──

    def health(self) -> Dict[str, str]:
        """检查各组件连接状态"""
        orch = self._orchestrator
        if orch is None:
            return {"status": "initializing", "vector_store": "pending", "graph_store": "pending", "llm": "pending"}
        return {
            "status": "healthy",
            "vector_store": "connected" if (orch.hybrid_retriever and orch.hybrid_retriever.vector_store) else "not_configured",
            "graph_store": "connected" if (orch.graph_rag and orch.graph_rag.graph_store) else "not_configured",
            "llm": "connected" if orch.llm_client else "not_configured",
        }

    def clear_history(self, conversation_id: Optional[str] = None):
        """清除对话历史"""
        if self._orchestrator:
            self._orchestrator.clear_history(conversation_id)

    def close(self):
        """释放资源（关闭图数据库连接等）"""
        if self._orchestrator:
            if self._orchestrator.graph_rag and hasattr(self._orchestrator.graph_rag.graph_store, "close"):
                self._orchestrator.graph_rag.graph_store.close()
            self._orchestrator.clear_history()
            self._orchestrator = None
