"""
文档分块器模块
============
支持多种分块策略：递归字符分割、语义分割、混合分割。
"""

from typing import List, Optional
from dataclasses import dataclass, field
import re


@dataclass
class DocumentChunk:
    """文档块数据结构"""
    chunk_id: str
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class TextChunker:
    """文本分块器"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        separators: Optional[List[str]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]

    def chunk(self, text: str, doc_id: str, metadata: Optional[dict] = None) -> List[DocumentChunk]:
        """递归字符分割"""
        chunks = []
        metadata = metadata or {}

        base_chunks = self._recursive_split(text, self.separators)
        texts = self._apply_overlap(base_chunks)

        for i, chunk_text in enumerate(texts):
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{i:04d}",
                doc_id=doc_id,
                content=chunk_text,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(texts),
                },
            )
            chunks.append(chunk)

        return chunks

    def _recursive_split(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """按结构从粗到细递归拆分为无重叠的基础块。"""
        separators = list(self.separators if separators is None else separators)
        target_size = self.chunk_size - self.chunk_overlap
        if len(text) <= target_size:
            return [text] if text.strip() else []

        selected_index = -1
        selected_separator = ""
        for index, separator in enumerate(separators):
            if separator == "" or separator in text:
                selected_index = index
                selected_separator = separator
                break

        if selected_index < 0 or selected_separator == "":
            return self._hard_split(text)

        remaining = separators[selected_index + 1:]
        parts = self._split_preserving_separator(text, selected_separator)
        units: List[str] = []
        for part in parts:
            if not part:
                continue
            if len(part) > target_size:
                units.extend(self._recursive_split(part, remaining))
            else:
                units.append(part)
        return self._merge_splits(units)

    @staticmethod
    def _split_preserving_separator(text: str, separator: str) -> List[str]:
        """在每个片段末尾保留匹配到的分隔符。"""
        if not separator:
            return [text]
        parts: List[str] = []
        start = 0
        while start < len(text):
            index = text.find(separator, start)
            if index < 0:
                parts.append(text[start:])
                break
            end = index + len(separator)
            parts.append(text[start:end])
            start = end
        return parts

    def _merge_splits(self, parts: List[str]) -> List[str]:
        """在有效载荷上限内合并递归产生的小片段。"""
        target_size = self.chunk_size - self.chunk_overlap
        output: List[str] = []
        current = ""
        for part in parts:
            if not part:
                continue
            if current and len(current) + len(part) > target_size:
                if current.strip():
                    output.append(current)
                current = ""
            if len(part) > target_size:
                if current.strip():
                    output.append(current)
                    current = ""
                output.extend(self._hard_split(part))
            else:
                current += part
        if current.strip():
            output.append(current)
        return output

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """给基础块添加前一段尾部上下文，同时保持最终大小上限。"""
        if not chunks or self.chunk_overlap == 0:
            return [item for item in chunks if item.strip()]
        output: List[str] = []
        history = ""
        for chunk in chunks:
            if not chunk.strip():
                history += chunk
                continue
            prefix = history[-self.chunk_overlap:] if history else ""
            combined = prefix + chunk
            if len(combined) > self.chunk_size:
                combined = combined[-self.chunk_size:]
            output.append(combined)
            history += chunk
        return output

    def _hard_split(self, text: str) -> List[str]:
        """按有效载荷大小硬切；统一 overlap 在最终阶段添加。"""
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        while start < len(text):
            end = min(start + step, len(text))
            chunks.append(text[start:end])
            start = end
        return chunks


class SemanticChunker:
    """语义分块器——基于段落/主题边界分割"""

    def __init__(self, max_chunk_size: int = 1024, min_chunk_size: int = 128):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, doc_id: str, metadata: Optional[dict] = None) -> List[DocumentChunk]:
        """按段落和主题分割"""
        metadata = metadata or {}
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 1 <= self.max_chunk_size:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{chunk_idx:04d}",
                        doc_id=doc_id,
                        content=current,
                        metadata={**metadata, "chunk_index": chunk_idx},
                    ))
                    chunk_idx += 1
                current = para

        if current:
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_idx:04d}",
                doc_id=doc_id,
                content=current,
                metadata={**metadata, "chunk_index": chunk_idx},
            ))

        return chunks


def get_chunker(strategy: str = "recursive", **kwargs) -> TextChunker:
    """工厂函数：获取分块器实例"""
    if strategy == "semantic":
        return SemanticChunker(**kwargs)
    return TextChunker(**kwargs)
