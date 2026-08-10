"""AutoMemory Textual application."""

from __future__ import annotations

from textual.app import App

from .config import SecretStore
from .context import AutoMemoryContext
from .models import AutoMemoryConfig
from .paths import AutoMemoryPaths
from .runtime import AutoMemoryRuntime
from .screens import ChatScreen, EvaluationScreen, HelpScreen, KnowledgeScreen, SettingsScreen
from .services import ChatService, DiagnosticsService, EvaluationService, KnowledgeService, MinerUService, WebCaptureService
from .storage import StateRepository


class AutoMemoryApp(App):
    """Local-first terminal workspace for chat, knowledge, and evaluation."""

    TITLE = "AutoMemory"
    SUB_TITLE = "Local multimodal memory workspace"
    CSS_PATH = "styles.tcss"
    ENABLE_COMMAND_PALETTE = True
    SCREENS = {
        "chat": ChatScreen,
        "knowledge": KnowledgeScreen,
        "evaluation": EvaluationScreen,
        "settings": SettingsScreen,
        "help": HelpScreen,
    }
    BINDINGS = [
        ("1", "workspace('chat')", "Chat"),
        ("2", "workspace('knowledge')", "Knowledge"),
        ("3", "workspace('evaluation')", "Evaluation"),
        ("4", "workspace('settings')", "Settings"),
        ("5", "workspace('help')", "Help"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, *, paths: AutoMemoryPaths | None = None) -> None:
        super().__init__()
        paths = paths or AutoMemoryPaths.resolve()
        state = StateRepository(paths.state_db)
        config = AutoMemoryConfig.from_dict(state.load_settings())
        secrets = SecretStore()
        runtime = AutoMemoryRuntime(paths, config, secrets)
        try:
            runtime.start()
        except Exception:
            state.close()
            raise
        self.ctx = AutoMemoryContext(
            paths=paths,
            config=config,
            secrets=secrets,
            state=state,
            runtime=runtime,
            chat=ChatService(runtime, state),
            knowledge=KnowledgeService(runtime, state),
            evaluation=EvaluationService(runtime, state),
            web=WebCaptureService(),
            mineru=MinerUService(),
            diagnostics=DiagnosticsService(paths, state, runtime, secrets),
        )

    def on_mount(self) -> None:
        self.push_screen("chat")

    def action_workspace(self, name: str) -> None:
        if name not in self.SCREENS:
            return
        if isinstance(self.screen, self.SCREENS[name]):
            return
        self.switch_screen(name)

    def on_unmount(self) -> None:
        self.ctx.runtime.close()
        self.ctx.state.close()
