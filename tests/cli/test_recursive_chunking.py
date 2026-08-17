from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.config import AutoMemoryConfig
from agentic_rag.cli.errors import CancelledError, ConfigurationError
from agentic_rag.cli.models import ParsedDocument
from agentic_rag.cli.paths import AutoMemoryPaths
from agentic_rag.cli.services.ingestion import IngestionService


def make_service(tmp_path: Path) -> IngestionService:
    return IngestionService(
        knowledge=None,
        paths=AutoMemoryPaths.resolve(tmp_path),
        config=AutoMemoryConfig(rag_mode="balanced"),
    )


def test_cli_recursive_chunking_preserves_pages_and_global_sequence(tmp_path: Path):
    service = make_service(tmp_path)
    parsed = ParsedDocument(
        "demo",
        [
            {"page": 2, "text": "A" * 900},
            {"page": 5, "text": "B" * 900},
        ],
    )

    chunks = service._chunk(parsed, "doc_demo", "demo.pdf", "kb_demo", CancellationToken())

    assert len(chunks) >= 4
    assert [item.sequence for item in chunks] == list(range(len(chunks)))
    assert [item.id for item in chunks] == [f"doc_demo_chunk_{index:05d}" for index in range(len(chunks))]
    assert {item.page for item in chunks if set(item.text) == {"A"}} == {2}
    assert {item.page for item in chunks if set(item.text) == {"B"}} == {5}
    assert all(item.metadata["source"] == "demo.pdf" for item in chunks)
    assert all(item.metadata["category_id"] == "kb_demo" for item in chunks)
    assert all(item.metadata["chunker_version"] == "recursive-v2" for item in chunks)
    assert all("chunk_index" in item.metadata and "total_chunks" in item.metadata for item in chunks)


def test_cli_recursive_chunking_rejects_all_blank_pages(tmp_path: Path):
    service = make_service(tmp_path)
    parsed = ParsedDocument("blank", [{"page": 1, "text": " \n\n "}])

    with pytest.raises(ConfigurationError, match="no readable text chunks"):
        service._chunk(parsed, "doc_blank", "blank.txt", "default", CancellationToken())


def test_cli_recursive_chunking_propagates_cancellation(tmp_path: Path):
    service = make_service(tmp_path)
    parsed = ParsedDocument("demo", [{"page": 1, "text": "content" * 200}])
    cancel = CancellationToken()
    cancel.cancel()

    with pytest.raises(CancelledError):
        service._chunk(parsed, "doc_demo", "demo.txt", "default", cancel)
