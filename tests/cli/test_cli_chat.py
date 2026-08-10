from __future__ import annotations

from io import StringIO
from pathlib import Path

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.config import AutoMemoryConfig
from agentic_rag.cli.models import EventKind, OutputEvent
from agentic_rag.cli.services.chat import DirectChatService
from agentic_rag.cli.storage import StateRepository
from agentic_rag.cli.terminal import PlainTerminal


class FakeLLM:
    def stream_chat(self, messages, cancel):
        assert all("Evidence:" not in str(message.get("content")) for message in messages)
        yield "cloud "
        yield "answer"


def test_direct_chat_streams_without_retrieval_and_persists(tmp_path: Path) -> None:
    state = StateRepository(tmp_path / "state.db", tmp_path / "backups")
    conversation = state.ensure_active_conversation()
    service = DirectChatService(state, FakeLLM(), AutoMemoryConfig())
    stdout, stderr = StringIO(), StringIO()
    output = PlainTerminal(stdout=stdout, stderr=stderr)
    result = service.stream(conversation, "hello", output, CancellationToken())
    assert result["answer"] == "cloud answer"
    assert stdout.getvalue() == "cloud answer\n"
    messages = state.list_messages(conversation)
    assert [message["mode"] for message in messages] == ["direct", "direct"]
    assert messages[-1]["status"] == "complete"
    state.close()


def test_plain_terminal_keeps_progress_out_of_stdout() -> None:
    stdout, stderr = StringIO(), StringIO()
    output = PlainTerminal(stdout=stdout, stderr=stderr, interactive=False)
    output.emit(OutputEvent(kind=EventKind.PROGRESS, text="working"))
    assert stdout.getvalue() == ""
