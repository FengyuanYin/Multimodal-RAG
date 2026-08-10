from __future__ import annotations

import pytest

from agentic_rag.cli.errors import UsageError
from agentic_rag.cli.models import InputKind
from agentic_rag.cli.router import CommandRouter, tokenize_windows


def test_only_exact_s_prefix_enables_retrieval() -> None:
    router = CommandRouter()
    assert router.parse("hello").kind == InputKind.DIRECT_CHAT
    assert router.parse("search this").kind == InputKind.DIRECT_CHAT
    assert router.parse("/search this").kind == InputKind.COMMAND
    assert router.parse("/settings").kind == InputKind.COMMAND
    assert router.parse("/s question").kind == InputKind.RAG_CHAT
    assert router.parse("/s\tquestion").question == "question"
    assert router.parse("  /s question").kind == InputKind.DIRECT_CHAT
    assert router.parse("/something").kind == InputKind.COMMAND


def test_windows_tokenizer_preserves_paths_and_quotes() -> None:
    assert tokenize_windows('/add "C:\\My Docs\\paper.pdf" --vlm') == ["/add", "C:\\My Docs\\paper.pdf", "--vlm"]
    with pytest.raises(UsageError):
        tokenize_windows('/add "C:\\unfinished')
