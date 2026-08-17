from __future__ import annotations

from io import StringIO
from pathlib import Path

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.config import AutoMemoryConfig
from agentic_rag.cli.models import EventKind, OutputEvent, RetrievalHit, RetrievalResult
from agentic_rag.cli.services.chat import DirectChatService, GroundedChatService
from agentic_rag.cli.storage import StateRepository
from agentic_rag.cli.terminal import PlainTerminal


class FakeLLM:
    def stream_chat(self, messages, cancel):
        assert all("Evidence:" not in str(message.get("content")) for message in messages)
        yield "cloud "
        yield "answer"


class CaptureGroundedLLM:
    def __init__(self):
        self.messages = None

    def stream_chat(self, messages, cancel):
        self.messages = messages
        yield "Visible evidence [1]"


class VisualRetriever:
    def search(self, *_args):
        hit = RetrievalHit("chunk", "doc", "Doc", "text evidence", 1, "text", 1.0)
        hit.visual_evidence = [{
            "media_id": "figure-1",
            "image_type": "structure_diagram",
            "representation": "mermaid",
            "content": "flowchart LR\nA --> B",
            "fallback_used": False,
            "uncertainty": "none",
            "cached": False,
        }]
        return RetrievalResult([hit], {"requested_mode": "advanced", "advanced_vlm": {"analyzed": 1}})


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


def test_grounded_chat_includes_post_rerank_visual_evidence_and_sources(tmp_path: Path) -> None:
    state = StateRepository(tmp_path / "state.db", tmp_path / "backups")
    conversation = state.ensure_active_conversation()
    llm = CaptureGroundedLLM()
    service = GroundedChatService(state, llm, AutoMemoryConfig(rag_mode="advanced"), VisualRetriever())
    output = PlainTerminal(stdout=StringIO(), stderr=StringIO())

    result = service.stream(conversation, "explain the diagram", output, CancellationToken())

    user_content = llm.messages[-1]["content"]
    assert "Post-rerank VLM evidence" in user_content
    assert "flowchart LR\nA --> B" in user_content
    assert result["sources"][0]["visual_evidence"][0]["media_id"] == "figure-1"
    state.close()


def test_plain_terminal_keeps_progress_out_of_stdout() -> None:
    stdout, stderr = StringIO(), StringIO()
    output = PlainTerminal(stdout=stdout, stderr=stderr, interactive=False)
    output.emit(OutputEvent(kind=EventKind.PROGRESS, text="working"))
    assert stdout.getvalue() == ""


def test_interactive_terminal_renders_and_closes_progress_bar() -> None:
    stdout, stderr = StringIO(), StringIO()
    output = PlainTerminal(stdout=stdout, stderr=stderr, interactive=True)
    output.emit(OutputEvent(kind=EventKind.PROGRESS, phase="embedding", completed=0, total=4))
    output.emit(OutputEvent(kind=EventKind.PROGRESS, phase="embedding", completed=2, total=4))
    output.emit(OutputEvent(kind=EventKind.PROGRESS, phase="embedding", completed=4, total=4))
    output.emit(OutputEvent(kind=EventKind.RESULT, text="Imported demo.pdf"))

    rendered = stderr.getvalue()
    assert "0/4   0%" in rendered
    assert "2/4  50%" in rendered
    assert "4/4 100%\n" in rendered
    assert stdout.getvalue() == "Imported demo.pdf\n"


def test_interactive_terminal_closes_progress_before_error() -> None:
    stdout, stderr = StringIO(), StringIO()
    output = PlainTerminal(stdout=stdout, stderr=stderr, interactive=True)
    output.emit(OutputEvent(kind=EventKind.PROGRESS, phase="embedding", completed=1, total=4))
    output.write_error(RuntimeError("failure"), debug=True)

    assert "1/4  25%\n[INTERNAL_ERROR]" in stderr.getvalue()
