"""Base workspace screen."""

from textual.screen import Screen


class WorkspaceScreen(Screen):
    def notify_error(self, exc: Exception) -> None:
        diagnostics = getattr(self.app.ctx, "diagnostics", None)
        if diagnostics:
            diagnostics.record_error(f"{type(exc).__name__}: {exc}")
        self.app.notify(str(exc), title="AutoMemory", severity="error", timeout=8)
