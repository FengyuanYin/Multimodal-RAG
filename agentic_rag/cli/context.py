"""Application context construction and atomic cloud-client reload."""

from __future__ import annotations

from dataclasses import dataclass, field

from .cloud import CohereRerankClient, MinerUClient, OpenAICompatibleClient, WebClient
from .config import AutoMemoryConfig, ConfigStore
from .credentials import CredentialStore
from .models import SearchResult
from .paths import AutoMemoryPaths
from .services import ConnectionTester, DiagnosticsService, DirectChatService, EvaluationService, GroundedChatService, IngestionService, RetrievalService
from .storage import KnowledgeRepository, StateRepository


@dataclass
class AppContext:
    paths: AutoMemoryPaths
    config_store: ConfigStore
    config: AutoMemoryConfig
    credentials: CredentialStore
    state: StateRepository
    knowledge: KnowledgeRepository
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
        context = cls(paths, config_store, config, credentials, state, knowledge)
        context.current_conversation = state.ensure_active_conversation()
        context.reload_services(config)
        return context

    def reload_services(self, config: AutoMemoryConfig | None = None) -> None:
        candidate = config or self.config_store.load()
        old_clients = [self.llm_client, self.embedding_client, self.vlm_client, self.reranker_client, self.web_client]
        llm_key = self.credentials.get(candidate.llm.credential_name)
        embedding_key = self.credentials.get(candidate.embedding.credential_name)
        vlm_key = self.credentials.get(candidate.vlm.credential_name)
        reranker_key = self.credentials.get(candidate.reranker.credential_name)
        self.llm_client = OpenAICompatibleClient(candidate.llm, llm_key, service="LLM") if llm_key else None
        self.embedding_client = OpenAICompatibleClient(candidate.embedding, embedding_key, service="Embedding") if embedding_key else None
        self.vlm_client = OpenAICompatibleClient(candidate.vlm, vlm_key, service="VLM") if vlm_key else None
        self.reranker_client = CohereRerankClient(candidate.reranker, reranker_key) if reranker_key else None
        self.web_client = WebClient()
        self.config = candidate
        self.retrieval = RetrievalService(self.knowledge, candidate, embedding_client=self.embedding_client, reranker_client=self.reranker_client)
        self.retrieval.rebuild()
        self.ingestion = IngestionService(self.knowledge, self.paths, candidate, embedding_client=self.embedding_client, vlm_client=self.vlm_client, state=self.state)
        self.direct_chat = DirectChatService(self.state, self.llm_client, candidate)
        self.grounded_chat = GroundedChatService(self.state, self.llm_client, candidate, self.retrieval)
        self.evaluation = EvaluationService(self.retrieval, self.state, self.paths.exports_dir)
        self.diagnostics = DiagnosticsService(self.paths, self.state, self.knowledge, self.credentials, candidate)
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
        for client in (self.llm_client, self.embedding_client, self.vlm_client, self.reranker_client, self.web_client):
            if client:
                try:
                    client.close()
                except Exception:
                    pass
        self.knowledge.close()
        self.state.close()
