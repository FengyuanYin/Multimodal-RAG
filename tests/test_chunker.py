from __future__ import annotations

import pytest

from agentic_rag.processing.chunker import TextChunker


def contents(chunker: TextChunker, text: str) -> list[str]:
    return [item.content for item in chunker.chunk(text, "doc")]


def test_recursive_split_preserves_paragraph_and_sentence_separators():
    text = "第一段保留段落边界。\n\n第二段第一句。第二段第二句。\n第三行继续。"
    chunks = contents(TextChunker(chunk_size=18, chunk_overlap=0), text)

    assert chunks
    assert all(0 < len(item) <= 18 for item in chunks)
    assert "".join(chunks) == text
    assert chunks[0].endswith("\n\n")
    assert any(item.rstrip().endswith("。") for item in chunks[:-1])


def test_oversized_segment_recurses_to_lower_level_separator():
    text = "没有段落但有多个句子。第二个句子仍然很长。第三个句子结束。"
    chunks = contents(TextChunker(chunk_size=16, chunk_overlap=0), text)

    assert len(chunks) >= 3
    assert "".join(chunks) == text
    assert all(len(item) <= 16 for item in chunks)


def test_hard_split_has_bounded_overlap_and_makes_progress():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = contents(TextChunker(chunk_size=10, chunk_overlap=3), text)

    assert chunks == ["abcdefg", "efghijklmn", "lmnopqrstu", "stuvwxyz"]
    assert chunks[0][-3:] == chunks[1][:3]
    assert chunks[1][-3:] == chunks[2][:3]
    assert chunks[2][-3:] == chunks[3][:3]
    assert all(len(item) <= 10 for item in chunks)


def test_overlap_zero_reconstructs_original_without_duplication():
    text = "alpha beta gamma delta epsilon"
    chunks = contents(TextChunker(chunk_size=12, chunk_overlap=0), text)

    assert "".join(chunks) == text
    assert chunks == contents(TextChunker(chunk_size=12, chunk_overlap=0), text)


def test_blank_text_returns_no_chunks():
    assert contents(TextChunker(chunk_size=10, chunk_overlap=2), " \n\n ") == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_parameters_are_rejected(chunk_size: int, chunk_overlap: int):
    with pytest.raises(ValueError):
        TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
