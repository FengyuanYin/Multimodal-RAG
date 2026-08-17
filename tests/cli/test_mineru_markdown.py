from __future__ import annotations

import io
import json
import zipfile

from agentic_rag.cli.cloud.mineru import MinerUClient


def _archive(include_markdown: bool = True) -> bytes:
    stream = io.BytesIO()
    content = [{"page_idx": 0, "text": "page evidence", "img_path": "images/figure.png", "img_caption": ["Figure 1"]}]
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("result_content_list.json", json.dumps(content))
        bundle.writestr("images/figure.png", b"image-bytes")
        if include_markdown:
            bundle.writestr("paper.md", "# Paper\n\nHEAD\n\n![](images/figure.png)\n\nTAIL")
    return stream.getvalue()


def _normalize(archive: bytes):
    client = object.__new__(MinerUClient)
    return client.normalize_archive(archive, "paper.pdf", "mineru_official")


def test_mineru_preserves_complete_markdown_and_media_reference() -> None:
    parsed = _normalize(_archive())
    assert parsed.markdown == "# Paper\n\nHEAD\n\n![](images/figure.png)\n\nTAIL"
    assert parsed.markdown_source == "mineru_original"
    assert parsed.markdown_media_refs == [{"reference": "images/figure.png", "media_id": "figure_1_p1"}]


def test_mineru_generates_markdown_when_archive_has_only_pages() -> None:
    parsed = _normalize(_archive(False))
    assert parsed.markdown_source == "generated"
    assert "# paper.pdf" in parsed.markdown
    assert "page evidence" in parsed.markdown
