"""Isolated in-process RAG runtime for AutoMemory."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from .config import SecretStore, to_project_settings, validate_config
from .models import AutoMemoryConfig, DiagnosticItem
from .paths import AutoMemoryPaths


@dataclass
class RuntimeHealth:
    items: list[DiagnosticItem]

    @property
    def status(self) -> str:
        if any(item.status == "error" for item in self.items):
            return "error"
        if any(item.status == "degraded" for item in self.items):
            return "degraded"
        return "ok"


class AutoMemoryRuntime:
    def __init__(self, paths: AutoMemoryPaths, config: AutoMemoryConfig, secrets: SecretStore) -> None:
        self.paths = paths
        self.config = validate_config(config)
        self.secrets = secrets
        self.settings = None
        self.orchestrator = None
        self._lock = RLock()

    def start(self) -> RuntimeHealth:
        with self._lock:
            if self.orchestrator is not None:
                return self.health()
            self.settings = to_project_settings(self.config, self.secrets, self.paths)
            self.orchestrator = self._build_lightweight(self.settings)
            return self.health()

    @staticmethod
    def _build_lightweight(cfg):
        from agentic_rag.core.orchestrator import AgenticOrchestrator
        from agentic_rag.memory.graph_store import GraphStoreFactory
        from agentic_rag.memory.knowledge_repository import KnowledgeRepository
        from agentic_rag.memory.media_store import MediaRegistry
        from agentic_rag.rag.graph_rag import GraphRAGEngine
        from agentic_rag.rag.hybrid_retriever import HybridRetriever
        from agentic_rag.rag.standard_rag import StandardRAGEngine

        llm_client = None
        if cfg.llm_api_key:
            from openai import OpenAI
            llm_client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
        vlm_client = None
        if cfg.vlm_api_key:
            from openai import OpenAI
            vlm_client = OpenAI(api_key=cfg.vlm_api_key, base_url=cfg.vlm_base_url or cfg.llm_base_url)
        elif llm_client and cfg.vlm_model:
            vlm_client = llm_client

        repository = KnowledgeRepository(cfg.knowledge_db_path)
        graph_store = GraphStoreFactory.create(db_type="networkx") if cfg.graph_db_type == "networkx" else None
        media_store = MediaRegistry(cfg.media_store_path, auto_save=True, max_memory_bytes=cfg.media_max_memory_mb * 1024 * 1024)
        media_store.load()
        retriever = HybridRetriever(knowledge_repository=repository, graph_store=graph_store)
        retriever.media_store = media_store

        # Optional vector components are initialized only when explicitly configured.
        if getattr(cfg, "embedding_model", ""):
            try:
                from agentic_rag.processing.embedders import EmbedderFactory
                from agentic_rag.memory.vector_store import VectorStoreFactory
                retriever.embedder = EmbedderFactory.create(
                    provider="bge" if "bge" in cfg.embedding_model.lower() else "openai",
                    model_name=cfg.embedding_model, device=cfg.embedding_device,
                    dim=cfg.embedding_dim, api_key=cfg.llm_api_key,
                )
                retriever.vector_store = VectorStoreFactory.create(
                    db_type=cfg.vector_db_type, collection_name="default",
                    embedding_dim=cfg.embedding_dim, persist_dir=cfg.vector_db_path,
                    host=cfg.qdrant_host, port=cfg.qdrant_port,
                )
            except Exception:
                retriever.embedder = None
                retriever.vector_store = None
        retriever.rebuild_from_repository()
        standard = StandardRAGEngine(
            retriever=retriever, embedder=retriever.embedder, llm_client=llm_client,
            llm_model=cfg.llm_model, vlm_client=vlm_client, vlm_model=cfg.vlm_model,
        )
        graph_rag = GraphRAGEngine(graph_store=graph_store, vector_store=retriever.vector_store, embedder=retriever.embedder, llm_client=llm_client, llm_model=cfg.llm_model)
        orch = AgenticOrchestrator(
            standard_rag=standard, graph_rag=graph_rag, hybrid_retriever=retriever,
            llm_client=llm_client, llm_model=cfg.llm_model,
            vlm_client=vlm_client, vlm_model=cfg.vlm_model,
            media_store=media_store, enable_multimodal=cfg.enable_multimodal_retrieval,
        )
        orch.reranker = None
        if getattr(cfg, "reranker_model", ""):
            try:
                from agentic_rag.processing.reranker import RerankerFactory
                orch.reranker = RerankerFactory.create("bge", cfg.reranker_model, cfg.reranker_device)
            except Exception:
                orch.reranker = None
        orch.knowledge_repository = repository
        return orch

    @property
    def repository(self):
        return getattr(self.orchestrator, "knowledge_repository", None)

    @property
    def retriever(self):
        return getattr(self.orchestrator, "hybrid_retriever", None)

    def health(self) -> RuntimeHealth:
        orch = self.orchestrator
        if orch is None:
            return RuntimeHealth([DiagnosticItem("runtime", "error", "not started")])
        items = [
            DiagnosticItem("runtime", "ok", "local in-process"),
            DiagnosticItem("knowledge", "ok" if self.repository and self.repository.integrity_check() == "ok" else "error", str(self.paths.knowledge_db)),
            DiagnosticItem("llm", "ok" if orch.llm_client else "degraded", self.config.llm_model if orch.llm_client else "API key not configured"),
            DiagnosticItem("vector", "ok" if self.retriever and self.retriever.vector_store else "degraded", "configured" if self.retriever and self.retriever.vector_store else "keyword search available"),
            DiagnosticItem("reranker", "ok" if getattr(orch, "reranker", None) else "degraded", "configured" if getattr(orch, "reranker", None) else "retrieval scores used directly"),
            DiagnosticItem("vlm", "ok" if orch.vlm_client else "degraded", self.config.vlm_model or "not configured"),
        ]
        return RuntimeHealth(items)

    def reload(self, config: AutoMemoryConfig, secrets: SecretStore) -> RuntimeHealth:
        candidate = AutoMemoryRuntime(self.paths, config, secrets)
        health = candidate.start()
        if health.status == "error":
            candidate.close()
            raise RuntimeError("candidate AutoMemory runtime failed health validation")
        with self._lock:
            old = self.orchestrator
            self.config, self.secrets, self.settings, self.orchestrator = config, secrets, candidate.settings, candidate.orchestrator
            candidate.orchestrator = None
        self._close_orchestrator(old)
        return health

    @staticmethod
    def _close_orchestrator(orch: Any) -> None:
        if not orch:
            return
        repository = getattr(orch, "knowledge_repository", None)
        if repository:
            repository.close()
        graph_store = getattr(getattr(orch, "graph_rag", None), "graph_store", None)
        if graph_store and hasattr(graph_store, "close"):
            graph_store.close()

    def close(self) -> None:
        with self._lock:
            orch, self.orchestrator = self.orchestrator, None
        self._close_orchestrator(orch)
