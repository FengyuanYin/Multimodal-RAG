"""Safe settings and process-local secret workspace."""

from dataclasses import fields

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Select, Static, Switch

from ..config import safe_config_payload, validate_config
from ..models import AutoMemoryConfig
from ..widgets import WorkspaceNav
from .base import WorkspaceScreen


class SettingsScreen(WorkspaceScreen):
    def compose(self) -> ComposeResult:
        config = self.app.ctx.config
        yield WorkspaceNav()
        with Vertical(classes="workspace"):
            yield Label("Settings", classes="title")
            with VerticalScroll(classes="settings-form"):
                yield Label("LLM", classes="section-title")
                yield Input(value=config.llm_model, placeholder="Model", id="setting-llm-model")
                yield Input(value=config.llm_base_url, placeholder="OpenAI-compatible Base URL", id="setting-llm-url")
                yield Input(placeholder="Runtime-only API key (never saved)", password=True, id="secret-llm")
                yield Label("Retrieval", classes="section-title")
                yield Select([("Keyword", "keyword"), ("Vector", "vector"), ("Hybrid", "hybrid"), ("Multimodal", "multimodal")], value=config.retrieval_mode, id="setting-retrieval", allow_blank=False)
                yield Input(value=str(config.top_k), placeholder="Top K", id="setting-top-k", type="integer")
                yield Input(value=config.embedding_model, placeholder="Embedding model (empty = keyword-only)", id="setting-embedding")
                yield Input(value=config.reranker_model, placeholder="Optional reranker model", id="setting-reranker")
                yield Input(value=str(config.chunk_size), placeholder="Chunk size", id="setting-chunk-size", type="integer")
                yield Input(value=str(config.chunk_overlap), placeholder="Chunk overlap", id="setting-chunk-overlap", type="integer")
                with Horizontal(classes="form-row"):
                    yield Label("Build knowledge graph")
                    yield Switch(value=config.build_graph, id="setting-build-graph")
                yield Label("Vision model", classes="section-title")
                yield Input(value=config.vlm_model, placeholder="VLM model", id="setting-vlm-model")
                yield Input(value=config.vlm_base_url, placeholder="VLM Base URL (empty = LLM endpoint)", id="setting-vlm-url")
                yield Input(placeholder="Runtime-only VLM key", password=True, id="secret-vlm")
                yield Label("MinerU and Web", classes="section-title")
                yield Select([("Official MinerU", "official"), ("Self-hosted MinerU", "selfhost")], value=config.mineru_mode, id="setting-mineru-mode", allow_blank=False)
                yield Input(value=config.mineru_url, placeholder="Self-hosted MinerU endpoint", id="setting-mineru-url")
                yield Input(placeholder="Runtime-only MinerU key", password=True, id="secret-mineru")
                yield Select([("DuckDuckGo", "duckduckgo"), ("Tavily", "tavily")], value=config.web_provider, id="setting-web-provider", allow_blank=False)
                yield Input(placeholder="Runtime-only Tavily key", password=True, id="secret-tavily")
                with Horizontal(classes="form-row"):
                    yield Label("Use enabled memories")
                    yield Switch(value=config.memory_enabled, id="setting-memory")
                with Horizontal(classes="form-row"):
                    yield Button("Save & reload", id="save-settings", variant="primary")
                    yield Button("Test health", id="test-settings")
                yield Static(self._secret_status(), id="settings-status", classes="status-line")

    def _secret_status(self) -> str:
        configured = [name for name in self.app.ctx.secrets.ENV_NAMES if self.app.ctx.secrets.configured(name)]
        return "Configured secrets: " + (", ".join(configured) if configured else "none")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "test-settings":
            report = self.app.ctx.runtime.health()
            self.query_one("#settings-status", Static).update(" | ".join(f"{item.name}: {item.status}" for item in report.items))
            return
        if event.button.id != "save-settings":
            return
        try:
            current = self.app.ctx.config
            updated = AutoMemoryConfig.from_dict(current.to_safe_dict())
            updated.llm_model = self.query_one("#setting-llm-model", Input).value.strip()
            updated.llm_base_url = self.query_one("#setting-llm-url", Input).value.strip()
            updated.retrieval_mode = str(self.query_one("#setting-retrieval", Select).value)
            updated.top_k = int(self.query_one("#setting-top-k", Input).value)
            updated.embedding_model = self.query_one("#setting-embedding", Input).value.strip()
            updated.reranker_model = self.query_one("#setting-reranker", Input).value.strip()
            updated.chunk_size = int(self.query_one("#setting-chunk-size", Input).value)
            updated.chunk_overlap = int(self.query_one("#setting-chunk-overlap", Input).value)
            updated.build_graph = self.query_one("#setting-build-graph", Switch).value
            updated.vlm_model = self.query_one("#setting-vlm-model", Input).value.strip()
            updated.vlm_base_url = self.query_one("#setting-vlm-url", Input).value.strip()
            updated.mineru_mode = str(self.query_one("#setting-mineru-mode", Select).value)
            updated.mineru_url = self.query_one("#setting-mineru-url", Input).value.strip()
            updated.web_provider = str(self.query_one("#setting-web-provider", Select).value)
            updated.memory_enabled = self.query_one("#setting-memory", Switch).value
            validate_config(updated)
            for name, selector in (("llm_api_key", "#secret-llm"), ("vlm_api_key", "#secret-vlm"), ("mineru_api_key", "#secret-mineru"), ("tavily_api_key", "#secret-tavily")):
                value = self.query_one(selector, Input).value
                if value:
                    self.app.ctx.secrets.set(name, value)
                    self.query_one(selector, Input).value = ""
            self.app.ctx.state.save_settings(safe_config_payload(updated))
            self.app.ctx.config = updated
            self.app.ctx.runtime.reload(updated, self.app.ctx.secrets)
            self.query_one("#settings-status", Static).update("Saved safe settings; runtime reloaded. " + self._secret_status())
        except Exception as exc:
            self.notify_error(exc)
