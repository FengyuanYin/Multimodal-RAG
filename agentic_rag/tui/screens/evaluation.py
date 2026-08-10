"""Retrieval evaluation workspace."""

from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, Select, Static

from ..events import CancelToken, JobProgress
from ..widgets import WorkspaceNav
from .base import WorkspaceScreen


class EvaluationScreen(WorkspaceScreen):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_token: CancelToken | None = None
        self.last_result: dict | None = None

    def compose(self) -> ComposeResult:
        yield WorkspaceNav()
        with Vertical(classes="workspace"):
            yield Label("Evaluation", classes="title")
            with Horizontal(classes="form-row"):
                yield Input(placeholder="Evaluation dataset JSON path", id="evaluation-path")
                yield Select([("Keyword", "keyword"), ("Vector", "vector"), ("Hybrid", "hybrid"), ("Multimodal", "multimodal")], value="keyword", id="evaluation-mode", allow_blank=False)
                yield Input(value="5", placeholder="Top K", id="evaluation-top-k", type="integer")
                yield Button("Run", id="run-evaluation", variant="primary")
                yield Button("Cancel", id="cancel-evaluation", variant="warning", disabled=True)
                yield Button("Export", id="export-evaluation", disabled=True)
            yield Static("Load a JSON dataset containing query and expected fields.", id="evaluation-status", classes="status-line")
            yield DataTable(id="evaluation-results", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        self.query_one(DataTable).add_columns("Metric", "Value")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-evaluation":
            path = Path(self.query_one("#evaluation-path", Input).value.strip())
            mode = str(self.query_one("#evaluation-mode", Select).value)
            try:
                top_k = int(self.query_one("#evaluation-top-k", Input).value)
            except ValueError:
                self.notify_error(ValueError("Top K must be an integer"))
                return
            self.cancel_token = CancelToken()
            self.query_one("#cancel-evaluation", Button).disabled = False
            self.run_evaluation(path, top_k, mode)
        elif event.button.id == "cancel-evaluation" and self.cancel_token:
            self.cancel_token.cancel()
        elif event.button.id == "export-evaluation" and self.last_result:
            name = f"evaluation-{self.last_result['run_id']}.json"
            destination = self.app.ctx.paths.exports_dir / name
            try:
                self.app.ctx.evaluation.export(self.last_result, destination, self.app.ctx.paths.exports_dir)
                self.query_one("#evaluation-status", Static).update(f"Exported: {destination}")
            except Exception as exc:
                self.notify_error(exc)

    def progress(self, event) -> None:
        if isinstance(event, JobProgress):
            suffix = f" ({event.completed}/{event.total})" if event.total else ""
            self.query_one("#evaluation-status", Static).update(f"{event.phase}: {event.message}{suffix}")

    def finish(self, result: dict | None, error: Exception | None = None) -> None:
        self.cancel_token = None
        self.query_one("#cancel-evaluation", Button).disabled = True
        if error:
            self.query_one("#evaluation-status", Static).update("Evaluation failed or cancelled")
            self.notify_error(error)
            return
        self.last_result = result
        table = self.query_one(DataTable)
        table.clear()
        for key, value in result["summary"].items():
            table.add_row(key, str(value))
        self.query_one("#export-evaluation", Button).disabled = False
        self.query_one("#evaluation-status", Static).update(f"Completed {result['count']} cases")

    @work(thread=True, exclusive=True, group="evaluation")
    def run_evaluation(self, path: Path, top_k: int, mode: str) -> None:
        try:
            result = self.app.ctx.evaluation.run(path, top_k, mode, emit=lambda event: self.app.call_from_thread(self.progress, event), cancel=self.cancel_token)
            self.app.call_from_thread(self.finish, result)
        except Exception as exc:
            self.app.call_from_thread(self.finish, None, exc)
