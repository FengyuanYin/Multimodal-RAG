"""Keyboard-friendly workspace navigation."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Label


class WorkspaceNav(Vertical):
    DEFAULT_CSS = """
    WorkspaceNav { width: 24; height: 100%; border-right: solid $primary-darken-2; padding: 1; }
    WorkspaceNav .brand { text-style: bold; color: $accent; margin-bottom: 1; }
    WorkspaceNav Button { width: 100%; margin-bottom: 1; content-align: left middle; }
    """

    def compose(self) -> ComposeResult:
        yield Label("AutoMemory", classes="brand")
        yield Button("1  Chat", id="nav-chat", variant="primary")
        yield Button("2  Knowledge", id="nav-knowledge")
        yield Button("3  Evaluation", id="nav-evaluation")
        yield Button("4  Settings", id="nav-settings")
        yield Button("5  Help", id="nav-help")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.app.action_workspace(event.button.id.removeprefix("nav-"))
