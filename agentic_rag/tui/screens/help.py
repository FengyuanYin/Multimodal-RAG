"""Help, keyboard map, paths, and diagnostics."""

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Label, Static

from ..widgets import WorkspaceNav
from .base import WorkspaceScreen


class HelpScreen(WorkspaceScreen):
    def compose(self) -> ComposeResult:
        yield WorkspaceNav()
        with Vertical(classes="workspace"):
            yield Label("Help & diagnostics", classes="title")
            with VerticalScroll(classes="help-content"):
                yield Static(
                    "**Keyboard**\n\n"
                    "`1–5` switch workspaces · `Ctrl+Enter` send · `Esc` cancel · `Ctrl+Q` quit\n\n"
                    "**Chat modes**\n\nDirect does not retrieve knowledge. Knowledge RAG retrieves from the selected collection and returns sources.\n\n"
                    "**Secrets**\n\nAPI keys are read from environment variables or held only for this process. They are never written to SQLite or logs."
                )
                yield Static(id="paths")
                yield Button("Refresh diagnostics", id="refresh-diagnostics", variant="primary")
                yield Static(id="diagnostics")

    def on_mount(self) -> None:
        paths = self.app.ctx.paths
        self.query_one("#paths", Static).update(f"**Local paths**\n\nData: `{paths.root}`\n\nExports: `{paths.exports_dir}`\n\nLogs: `{paths.logs_dir}`")
        self.refresh_report()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh-diagnostics":
            self.refresh_report()

    def refresh_report(self) -> None:
        lines = [f"- **{item.status.upper()}** {item.name}: {item.detail}" for item in self.app.ctx.diagnostics.report()]
        errors = self.app.ctx.diagnostics.recent_errors()
        if errors:
            lines.extend(["", "**Recent sanitized errors**", *[f"- {item}" for item in errors]])
        self.query_one("#diagnostics", Static).update("\n".join(lines))
