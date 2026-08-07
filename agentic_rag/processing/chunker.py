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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]

    def chunk(self, text: str, doc_id: str, metadata: Optional[dict] = None) -> List[DocumentChunk]:
        """递归字符分割"""
        chunks = []
        metadata = metadata or {}

        # 递归分割
        texts = self._recursive_split(text)

        for i, chunk_text in enumerate(texts):
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{i:04d}",
                doc_id=doc_id,
                content=chunk_text.strip(),
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(texts),
                },
            )
            chunks.append(chunk)

        return chunks

    def _recursive_split(self, text: str) -> List[str]:
        """递归分割实现"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for sep in self.separators:
            if sep == "":
                # 按字符数硬切
                return self._hard_split(text)
            if sep in text:
                segments = text.split(sep)
                result = []
                current = ""
                for seg in segments:
                    candidate = current + sep + seg if current else seg
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            result.append(current)
                        current = seg
                if current:
                    result.append(current)
                # 如果结果块仍然太大，递归处理
                final = []
                for r in result:
                    if len(r) > self.chunk_size:
                        final.extend(self._recursive_split(r))
                    else:
                        final.append(r)
                return final

        return self._hard_split(text)

    def _hard_split(self, text: str) -> List[str]:
        """按字符数硬切，带重叠"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
            if start >= len(text):
                break
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