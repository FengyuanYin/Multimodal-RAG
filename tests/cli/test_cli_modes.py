from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from agentic_rag.cli.commands import register_all
from agentic_rag.cli.config import AutoMemoryConfig
from agentic_rag.cli.repl import run_repl
from agentic_rag.cli.router import CommandRouter
from agentic_rag.cli.terminal import PlainTerminal


def test_interactive_repl_shows_brand_but_pipe_path_does_not() -> None:
    router = CommandRouter()
    register_all(router)
    stdout = StringIO()
    terminal = PlainTerminal(stdin=StringIO("/exit\n"), stdout=stdout, stderr=StringIO(), interactive=True, color=False)
    ctx = SimpleNamespace(config=AutoMemoryConfig(), credentials=SimpleNamespace(configured=lambda name: False))
    assert run_repl(ctx, terminal, router) == 0
    assert "AutoMemory" in stdout.getvalue()
    assert "/setup" in stdout.getvalue()
    assert "\x1b[" not in stdout.getvalue()
