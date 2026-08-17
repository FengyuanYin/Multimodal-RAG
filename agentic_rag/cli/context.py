"""Application context construction and atomic cloud-client reload."""

from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger

from agentic_rag.memory.vector_store import VectorStoreFactory
from .cloud import CohereRerankClient, MinerUClient, OpenAICompatibleClient, WebClient
from .config import AutoMemoryConfig, ConfigStore
from .credentials import CredentialStore
from .models import SearchResult
from .paths import AutoMemoryPaths
from .services import AdvancedVisualRouter, ConnectionTester, ConservativeTokenEstimator, ContextBudgetService, DiagnosticsService, DirectChatService, DocumentArtifactService, DocumentContextBuilder, DocumentWorkspaceChatService, EvaluationService, GraphExportService, GraphRetrievalService, GroundedChatService, ImageAnalysisService, IndexPreparationService, IngestionService, LongResponseService, QueryRewriteService, RetrievalService, WorkspaceCompactionService, WorkspaceFileService
from .storage import KnowledgeRepository, StateRepository


@dataclass
class AppContext:
    paths: AutoMemoryPaths
    config_store: ConfigStore
    config: AutoMemoryConfig
    credentials: CredentialStore
    state: StateRepository
    knowledge: KnowledgeRepository
    vector_store: object | None = None
    llm_client: object | None = None
    embedding_client: object | None = None
    vlm_client: object | None = None
    reranker_client: object | None = None
    web_client: WebClient | None = None
    retrieval: RetrievalService | None = None
    ingestion: IngestionService | None = None
    direct_chat: DirectChatService | None = None
    grounded_chat: GroundedChatService | None = None
    evaluation: EvaluationService | None = None
    diagnostics: DiagnosticsService | None = None
    connectivity: ConnectionTester | None = None
    index_preparation: IndexPreparationService | None = None
    query_rewriter: QueryRewriteService | None = None
    graph_retriever: GraphRetrievalService | None = None
    graph_export: GraphExportService | None = None
    advanced_visual_router: AdvancedVisualRouter | None = None
    document_artifacts: object | None = None
    workspace_files: object | None = None
    document_workspace_chat: object | None = None
    current_conversation: str = ""
    search_results: list[SearchResult] = field(default_factory=list)
    last_trace: dict = field(default_factory=dict)

    @classmethod
    def create(cls, paths: AutoMemoryPaths) -> "AppContext":
        config_store = ConfigStore(paths.config_file, paths.backups_dir)
        config = config_store.load()
        credentials = CredentialStore()
        state = StateRepository(paths.state_db, paths.backups_dir)
        knowledge = KnowledgeRepository(paths.knowledge_db, paths.backups_dir)
        if config.active_category not in {item["id"] for item in knowledge.list_knowledge_bases()}:
            payload = config.to_dict()
            payload["active_category"] = "default"
            config = AutoMemoryConfig.from_dict(payload)
            config_store.save(config)
        context = cls(paths, config_store, config, credentials, state, knowledge)
        context.current_conversation = state.ensure_active_conversation()
        context.reload_services(config)
        return context

    def reload_services(self, config: AutoMemoryConfig | None = None) -> None:
        candidate = config or self.config_store.load()
        old_clients = [self.llm_client, self.embedding_client, self.vlm_client, self.reranker_client, self.web_client, self.vector_store]
        llm_key = self.credentials.get(candidate.llm.credential_name)
        embedding_key = self.credentials.get(candidate.embedding.credential_name)
        vlm_key = self.credentials.get(candidate.vlm.credential_name)
        reranker_key = self.credentials.get(candidate.reranker.credential_name)
        self.llm_client = OpenAICompatibleClient(candidate.llm, llm_key, service="LLM") if llm_key else None
        self.embedding_client = OpenAICompatibleClient(candidate.embedding, embedding_key, service="Embedding") if embedding_key else None
        self.vlm_client = OpenAICompatibleClient(candidate.vlm, vlm_key, service="VLM") if vlm_key else None
        self.reranker_client = CohereRerankClient(candidate.reranker, reranker_key) if reranker_key else None
        self.web_client = WebClient()
        try:
            self.vector_store = VectorStoreFactory.create(
                db_type="milvus",
                collection_name=candidate.milvus_collection,
                uri=candidate.milvus_uri,
                database=candidate.milvus_database,
                token=self.credentials.get("milvus_token"),
                timeout_seconds=float(candidate.milvus_timeout_seconds),
            )
        except Exception as exc:
            self.vector_store = None
            logger.warning("Milvus vector storage unavailable: {}", type(exc).__name__)
        self.config = candidate
        self.query_rewriter = QueryRewriteService(self.llm_client)
        self.graph_retriever = GraphRetrievalService(self.knowledge)
        self.graph_export = GraphExportService(self.knowledge, self.paths.exports_dir)
        self.advanced_visual_router = AdvancedVisualRouter(self.knowledge, self.vlm_client)
        self.index_preparation = IndexPreparationService(self.knowledge, vector_store=self.vector_store, embedding_client=self.embedding_client, llm_client=self.llm_client, batch_delay_seconds=float(candidate.embedding_batch_delay_seconds))
        self.retrieval = RetrievalService(self.knowledge, candidate, vector_store=self.vector_store, embedding_client=self.embedding_client, reranker_client=self.reranker_client, query_rewriter=self.query_rewriter, graph_retriever=self.graph_retriever, visual_router=self.advanced_visual_router)
        self.retrieval.rebuild()
        estimator = ConservativeTokenEstimator()
        self.document_artifacts = DocumentArtifactService(self.knowledge, self.paths.knowledge_assets_dir)
        self.workspace_files = WorkspaceFileService(self.state.workspaces, self.paths.workspaces_dir, self.paths.exports_dir, estimator)
        budget = ContextBudgetService(candidate, estimator)
        builder = DocumentContextBuilder()
        images = ImageAnalysisService(self.knowledge, self.state.workspaces, self.workspace_files, self.vlm_client)
        compaction = WorkspaceCompactionService(self.llm_client, self.workspace_files, self.state.workspaces, candidate)
        long_responses = LongResponseService(candidate, self.workspace_files, estimator)
        self.document_workspace_chat = DocumentWorkspaceChatService(self.state, self.knowledge, self.document_artifacts, self.llm_client, builder, budget, compaction, self.workspace_files, images, long_responses, candidate)
        self.ingestion = IngestionService(self.knowledge, self.paths, candidate, vector_store=self.vector_store, embedding_client=self.embedding_client, vlm_client=self.vlm_client, state=self.state, index_preparation=self.index_preparation, artifact_service=self.document_artifacts)
        self.direct_chat = DirectChatService(self.state, self.llm_client, candidate)
        self.grounded_chat = GroundedChatService(self.state, self.llm_client, candidate, self.retrieval)
        self.evaluation = EvaluationService(self.retrieval, self.state, self.paths.exports_dir)
        self.diagnostics = DiagnosticsService(self.paths, self.state, self.knowledge, self.credentials, candidate, vector_store=self.vector_store)
        self.connectivity = ConnectionTester(self)
        for client in old_clients:
            if client and client not in {self.llm_client, self.embedding_client, self.vlm_client, self.reranker_client, self.web_client}:
                try:
                    client.close()
                except Exception:
                    pass

    def save_config(self, config: AutoMemoryConfig) -> None:
        self.config_store.save(config)
        self.reload_services(config)

    def mineru_client(self, mode: str | None = None) -> MinerUClient:
        selected = mode or self.config.mineru_mode
        key = self.credentials.get("mineru_api_key")
        if selected == "official":
            return MinerUClient("https://mineru.net/api/v4", key)
        return MinerUClient(self.config.mineru_url, key, allow_private=True)

    def close(self) -> None:
        for client in (self.llm_client, self.embedding_client, self.vlm_client, self.reranker_client, self.web_client, self.vector_store):
            if client:
                try:
                    client.close()
                except Exception:
                    pass
        self.knowledge.close()
        self.state.close()
