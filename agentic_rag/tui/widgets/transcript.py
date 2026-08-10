"""Streaming conversation transcript."""

from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static


class Transcript(VerticalScroll):
    DEFAULT_CSS = """
    Transcript { height: 1fr; padding: 1 2; }
    Transcript .message { width: 100%; margin-bottom: 1; padding: 1 2; }
    Transcript .user { background: $primary 18%; border-left: thick $primary; }
    Transcript .assistant { background: $surface-lighten-1; border-left: thick $accent; }
    Transcript .status { color: $text-muted; text-style: italic; }
    """

    def add_message(self, role: str, content: str, status: str = "") -> Markdown:
        label = "You" if role == "user" else "AutoMemory"
        suffix = f"  _{status}_" if status else ""
        widget = Markdown(f"**{label}**{suffix}\n\n{content}", classes=f"message {role}")
        self.mount(widget)
        self.scroll_end(animate=False)
        return widget

    def empty_state(self, text: str) -> None:
        self.remove_children()
        self.mount(Static(text, classes="status"))
