from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.commands.mineru import mineru
from agentic_rag.cli.commands.web import fetch
from agentic_rag.cli.models import CapturedPage, ParsedDocument
from agentic_rag.cli.terminal import PlainTerminal


class FakeState:
    def create_task(self, *_args, **_kwargs) -> str:
        return "task_test"

    def update_task(self, *_args, **_kwargs) -> None:
        pass


class FakeMinerUClient:
    def __init__(self) -> None:
        self.closed = False

    def parse_official(self, *_args, **_kwargs) -> ParsedDocument:
        return ParsedDocument("paper.pdf", [{"page": 1, "text": "evidence"}])

    def close(self) -> None:
        self.closed = True


class CapturingIngestion:
    def __init__(self) -> None:
        self.category = ""

    def ingest_parsed(self, _parsed, _source, category, _output, _cancel, _source_type):
        self.category = category
        return {"message": "Imported paper.pdf"}

    def ingest_web(self, _page, category, _output, _cancel):
        self.category = category
        return {"message": "Imported page"}


def _output() -> PlainTerminal:
    return PlainTerminal(stdout=StringIO(), stderr=StringIO(), interactive=False)


def test_mineru_defaults_to_active_knowledge_base(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")
    client = FakeMinerUClient()
    ingestion = CapturingIngestion()
    ctx = SimpleNamespace(
        config=SimpleNamespace(active_category="cat_skills", mineru_mode="official"),
        mineru_client=lambda _mode: client,
        state=FakeState(),
        ingestion=ingestion,
        retrieval=SimpleNamespace(rebuild=lambda _cancel: None),
    )

    mineru(ctx, [str(pdf)], _output(), CancellationToken(), None)

    assert ingestion.category == "cat_skills"
    assert client.closed


def test_mineru_explicit_category_still_overrides_active_base(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")
    ingestion = CapturingIngestion()
    ctx = SimpleNamespace(
        config=SimpleNamespace(active_category="cat_skills", mineru_mode="official"),
        mineru_client=lambda _mode: FakeMinerUClient(),
        state=FakeState(),
        ingestion=ingestion,
        retrieval=SimpleNamespace(rebuild=lambda _cancel: None),
    )

    mineru(ctx, [str(pdf), "--category", "default"], _output(), CancellationToken(), None)

    assert ingestion.category == "default"


def test_fetch_defaults_to_active_knowledge_base() -> None:
    ingestion = CapturingIngestion()
    page = CapturedPage("Example", "https://example.com", "captured text")
    ctx = SimpleNamespace(
        config=SimpleNamespace(active_category="cat_research"),
        search_results=[],
        web_client=SimpleNamespace(fetch=lambda _target, _cancel: page),
        ingestion=ingestion,
        retrieval=SimpleNamespace(rebuild=lambda _cancel: None),
    )

    fetch(ctx, ["https://example.com", "--yes"], _output(), CancellationToken(), None)

    assert ingestion.category == "cat_research"
