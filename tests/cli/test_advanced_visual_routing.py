from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.errors import CancelledError
from agentic_rag.cli.models import ChunkRecord, DocumentRecord, MediaRecord, RetrievalHit
from agentic_rag.cli.services.advanced_visual_router import AdvancedVisualRouter
from agentic_rag.cli.storage import KnowledgeRepository


class FakeVLM:
    profile_fingerprint = "vlm-profile"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def describe_image(self, content, mime_type, prompt, cancel):
        cancel.checkpoint()
        self.calls.append({"content": content, "mime_type": mime_type, "prompt": prompt})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_repo(tmp_path: Path, media_specs) -> KnowledgeRepository:
    repo = KnowledgeRepository(tmp_path / "knowledge.db", tmp_path / "backups")
    document = DocumentRecord("doc", "fp", "Doc", "doc.pdf", "pdf", "default", "mineru", 1, "ready")
    chunk = ChunkRecord("chunk", "doc", 1, 0, "evidence")
    media = []
    for media_id, media_type, raw in media_specs:
        path = tmp_path / f"{media_id}.png"
        path.write_bytes(raw)
        media.append(MediaRecord(
            media_id, "doc", 1, media_type, media_id,
            mime_type="image/png" if media_type == "image" else "text/markdown",
            checksum=hashlib.sha256(raw).hexdigest(), storage_path=str(path), quality="exact",
        ))
    repo.commit_document(document, [chunk], media)
    return repo


def hit(*media_ids: str, target_id: str = "chunk") -> RetrievalHit:
    return RetrievalHit(
        target_id, "doc", "Doc", "evidence", 1, "text", 1.0,
        {"q0:reference_graph": 1.0, "multimodal": 1.0},
        [{"media_id": media_id, "media_type": "image"} for media_id in media_ids],
    )


def test_routes_all_unique_images_once_and_reuses_persistent_cache(tmp_path: Path):
    repo = make_repo(tmp_path, [("chart", "image", b"chart"), ("structure", "image", b"structure"), ("photo", "image", b"photo")])
    vlm = FakeVLM([
        json.dumps({"image_type": "data_chart", "markdown_table": "| Series | Value |\n|---|---:|\n| A | 12 |", "mermaid": "", "description": "", "uncertainty": "none"}),
        "```json\n" + json.dumps({"image_type": "structure_diagram", "markdown_table": "", "mermaid": "flowchart LR\nA[Input] --> B[Output]", "description": "", "uncertainty": "arrow label unclear"}) + "\n```",
        json.dumps({"image_type": "other", "markdown_table": "", "mermaid": "", "description": "A product photograph on a desk.", "uncertainty": "brand unreadable"}),
    ])
    router = AdvancedVisualRouter(repo, vlm)
    hits = [hit("chart", "structure"), hit("chart", "photo", target_id="chunk-2")]

    report = router.enrich(hits, CancellationToken())

    assert len(vlm.calls) == 3
    assert report == {"eligible_images": 4, "unique_images": 3, "cache_hits": 0, "primary_calls": 3, "fallback_calls": 0, "analyzed": 3, "degraded": []}
    assert [item["image_type"] for item in hits[0].visual_evidence] == ["data_chart", "structure_diagram"]
    assert [item["media_id"] for item in hits[1].visual_evidence] == ["chart", "photo"]
    assert hits[0].visual_evidence[0]["representation"] == "markdown_table"
    assert hits[0].visual_evidence[1]["representation"] == "mermaid"
    assert hits[1].visual_evidence[1]["representation"] == "narrative"

    cached_hits = [hit("chart", "structure", "photo")]
    cached_vlm = FakeVLM([])
    cached_report = AdvancedVisualRouter(repo, cached_vlm).enrich(cached_hits, CancellationToken())
    assert cached_vlm.calls == []
    assert cached_report["cache_hits"] == 3
    assert cached_report["analyzed"] == 3
    assert all(item["cached"] for item in cached_hits[0].visual_evidence)
    repo.close()


def test_invalid_typed_output_uses_exactly_one_general_visual_fallback(tmp_path: Path):
    repo = make_repo(tmp_path, [("chart", "image", b"chart")])
    vlm = FakeVLM([
        json.dumps({"image_type": "data_chart", "markdown_table": "not a table", "mermaid": "", "description": "", "uncertainty": ""}),
        "The image appears to be a bar chart comparing three categories.",
    ])
    hits = [hit("chart")]

    report = AdvancedVisualRouter(repo, vlm).enrich(hits, CancellationToken())

    assert len(vlm.calls) == 2
    assert report["primary_calls"] == 1
    assert report["fallback_calls"] == 1
    assert hits[0].visual_evidence[0]["image_type"] == "fallback"
    assert hits[0].visual_evidence[0]["fallback_used"] is True
    assert hits[0].visual_evidence[0]["content"].startswith("The image")
    repo.close()


def test_non_image_and_bad_checksum_degrade_without_vlm_call(tmp_path: Path):
    repo = make_repo(tmp_path, [("bad", "image", b"bad"), ("table", "table", b"table")])
    repo._conn.execute("UPDATE media SET checksum='wrong' WHERE id='bad'")
    repo._conn.commit()
    vlm = FakeVLM([])
    hits = [hit("bad", "table")]

    report = AdvancedVisualRouter(repo, vlm).enrich(hits, CancellationToken())

    assert vlm.calls == []
    assert report["unique_images"] == 1
    assert report["analyzed"] == 0
    assert report["degraded"] == [{"media_id": "bad", "reason": "checksum_mismatch"}]
    repo.close()


def test_missing_vlm_only_degrades_when_an_eligible_image_exists(tmp_path: Path):
    repo = make_repo(tmp_path, [("image", "image", b"image")])
    with_image = AdvancedVisualRouter(repo, None).enrich([hit("image")], CancellationToken())
    without_image = AdvancedVisualRouter(repo, None).enrich([hit()], CancellationToken())

    assert with_image["unique_images"] == 1
    assert with_image["degraded"] == [{"media_id": "image", "reason": "vlm_not_configured"}]
    assert without_image["unique_images"] == 0
    assert without_image["degraded"] == []
    repo.close()


def test_cancelled_vlm_call_propagates(tmp_path: Path):
    repo = make_repo(tmp_path, [("image", "image", b"image")])
    vlm = FakeVLM([CancelledError("cancelled")])

    with pytest.raises(CancelledError):
        AdvancedVisualRouter(repo, vlm).enrich([hit("image")], CancellationToken())
    repo.close()
