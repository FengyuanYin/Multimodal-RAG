"""Knowledge ingestion, Web capture, and document management workspace."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Input, Label, Select, Static

from ..events import CancelToken
from ..widgets import WorkspaceNav
from .base import WorkspaceScreen


class KnowledgeScreen(WorkspaceScreen):
    def __init__(self) -> None:
        super().__init__()
        self.selected_document_id = ""
        self.cancel_token: CancelToken | None = None

    def compose(self) -> ComposeResult:
        yield WorkspaceNav()
        with Vertical(classes="workspace"):
            yield Label("Knowledge", classes="title")
            with Horizontal(classes="toolbar"):
                yield Select([("Default", "default")], value="default", id="knowledge-collection", allow_blank=False)
                yield Input(placeholder="New category", id="category-name")
                yield Button("Add category", id="add-category")
                yield Button("Delete category", id="delete-category", variant="error")
                yield Button("Refresh", id="refresh-documents")
                yield Button("Rebuild index", id="rebuild-index")
                yield Button("Cancel job", id="cancel-knowledge", variant="warning", disabled=True)
            with VerticalScroll(classes="knowledge-actions"):
                yield Label("Local files", classes="section-title")
                with Horizontal(classes="form-row"):
                    yield Input(placeholder="PDF, TXT, Markdown, image, CSV/XLSX path(s), separated by ;", id="local-paths")
                    yield Button("Import", id="import-local", variant="primary")
                yield Label("Web search and capture", classes="section-title")
                with Horizontal(classes="form-row"):
                    yield Input(placeholder="Search keywords", id="web-query")
                    yield Button("Search", id="search-web")
                with Horizontal(classes="form-row"):
                    yield Input(placeholder="https://example.com/article", id="web-url")
                    yield Button("Capture URL", id="capture-web", variant="primary")
                yield Label("MinerU PDF parsing", classes="section-title")
                with Horizontal(classes="form-row"):
                    yield Input(placeholder="Local PDF path", id="mineru-path")
                    yield Button("Parse & import", id="parse-mineru", variant="primary")
                yield Static("Ready", id="knowledge-status", classes="status-line")
            with Horizontal(classes="compact-actions"):
                yield Button("Show selected details", id="document-detail")
                yield Button("Delete selected document", id="delete-document", variant="error")
            yield DataTable(id="documents", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Title", "Type", "Parser", "Chunks", "Media", "Status")
        self.refresh_categories()
        self.refresh_documents()

    def refresh_categories(self) -> None:
        options = [(item.name, item.id) for item in self.app.ctx.state.list_categories()]
        select = self.query_one("#knowledge-collection", Select)
        select.set_options(options)
        select.value = options[0][1]

    def refresh_documents(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        for item in self.app.ctx.knowledge.list_documents():
            table.add_row(item.get("name", ""), item.get("source_type", ""), item.get("parser", ""), str(item.get("chunk_count", 0)), str(item.get("media_count", 0)), item.get("status", ""), key=item["id"])

    def set_status(self, text: str, error: Exception | None = None) -> None:
        self.cancel_token = None
        self.query_one("#cancel-knowledge", Button).disabled = True
        self.query_one("#knowledge-status", Static).update(text)
        if error:
            self.notify_error(error)
        else:
            self.refresh_documents()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action = event.button.id
        collection = str(self.query_one("#knowledge-collection", Select).value)
        if action == "cancel-knowledge" and self.cancel_token:
            self.cancel_token.cancel()
        elif action == "add-category":
            try:
                self.app.ctx.state.create_category(self.query_one("#category-name", Input).value)
                self.query_one("#category-name", Input).value = ""
                self.refresh_categories()
            except Exception as exc:
                self.notify_error(exc)
        elif action == "delete-category":
            try:
                if any(item.get("category_id") == collection for item in self.app.ctx.knowledge.list_documents()):
                    raise ValueError("delete or move documents in this category first")
                self.app.ctx.state.delete_category(collection)
                self.refresh_categories()
            except Exception as exc:
                self.notify_error(exc)
        elif action == "document-detail" and self.selected_document_id:
            detail = self.app.ctx.knowledge.document_detail(self.selected_document_id)
            self.query_one("#knowledge-status", Static).update(str(detail or "Document not found")[:1500])
        elif action == "delete-document" and self.selected_document_id:
            try:
                result = self.app.ctx.knowledge.delete_document(self.selected_document_id)
                self.selected_document_id = ""
                self.set_status(str(result.get("message") or result))
            except Exception as exc:
                self.notify_error(exc)
        elif action == "refresh-documents":
            self.refresh_documents()
        elif action == "rebuild-index":
            self.start_operation("rebuild", "", collection)
        elif action == "import-local":
            self.start_operation("local", self.query_one("#local-paths", Input).value, collection)
        elif action == "search-web":
            self.start_operation("search", self.query_one("#web-query", Input).value, collection)
        elif action == "capture-web":
            self.start_operation("capture", self.query_one("#web-url", Input).value, collection)
        elif action == "parse-mineru":
            self.start_operation("mineru", self.query_one("#mineru-path", Input).value, collection)

    def start_operation(self, action: str, value: str, collection: str) -> None:
        if self.cancel_token:
            return
        self.cancel_token = CancelToken()
        self.query_one("#cancel-knowledge", Button).disabled = False
        self.run_operation(action, value, collection, self.cancel_token)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "documents":
            self.selected_document_id = str(event.row_key.value)
            self.query_one("#knowledge-status", Static).update(f"Selected document: {self.selected_document_id}")

    @work(thread=True, exclusive=True, group="knowledge")
    def run_operation(self, action: str, value: str, collection: str, cancel: CancelToken) -> None:
        try:
            self.app.call_from_thread(self.query_one("#knowledge-status", Static).update, f"{action}: running…")
            if action == "rebuild":
                result = self.app.ctx.knowledge.rebuild_indexes()
            elif action == "local":
                paths = [Path(item.strip()) for item in value.split(";") if item.strip()]
                if not paths:
                    raise ValueError("enter at least one local file path")
                result = self.app.ctx.knowledge.ingest_local(paths, collection, cancel=cancel)
            elif action == "search":
                if not value.strip():
                    raise ValueError("enter search keywords")
                results = asyncio.run(self.app.ctx.web.search(value.strip(), self.app.ctx.config.web_provider, self.app.ctx.secrets.get("tavily_api_key"), cancel=cancel))
                if results:
                    self.app.call_from_thread(setattr, self.query_one("#web-url", Input), "value", results[0].url)
                result = {"message": " | ".join(f"{item.title}: {item.url}" for item in results[:5]) or "No results"}
            elif action == "capture":
                page = asyncio.run(self.app.ctx.web.fetch(value.strip(), cancel=cancel))
                result = self.app.ctx.knowledge.ingest_page(page, collection, cancel=cancel)
            else:
                path = Path(value.strip())
                if not path.is_file():
                    raise ValueError("enter an existing PDF path")
                if self.app.ctx.config.mineru_mode == "official":
                    parsed = asyncio.run(self.app.ctx.mineru.parse_official(path, self.app.ctx.secrets.get("mineru_api_key"), cancel=cancel))
                else:
                    parsed = asyncio.run(self.app.ctx.mineru.parse_selfhosted(path, self.app.ctx.config.mineru_url, self.app.ctx.secrets.get("mineru_api_key"), cancel=cancel))
                result = self.app.ctx.knowledge.ingest_external(parsed, str(path.resolve()), collection, cancel=cancel)
            self.app.call_from_thread(self.set_status, str(result.get("message") or result))
        except Exception as exc:
            self.app.call_from_thread(self.set_status, f"{action}: failed", exc)
