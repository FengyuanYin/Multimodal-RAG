"""
多模态解析器模块
==============
支持文本、图片、表格、PDF 等多模态文档的解析与统一表示。
"""

from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import base64
import io
import re
from loguru import logger


@dataclass
class DocumentChunk:
    """统一文档块"""
    chunk_id: str
    doc_id: str
    content: str
    modality: str = "text"  # text | image | table | mixed
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class ParsedDocument:
    """解析后的文档"""
    doc_id: str
    title: str
    modality: str
    chunks: List[DocumentChunk]
    raw_metadata: dict = field(default_factory=dict)


class TextParser:
    """文本解析器"""

    def parse(self, content: str, doc_id: str, metadata: Optional[dict] = None) -> ParsedDocument:
        metadata = metadata or {}
        chunk = DocumentChunk(
            chunk_id=f"{doc_id}_chunk_0000",
            doc_id=doc_id,
            content=content,
            modality="text",
            metadata=metadata,
        )
        return ParsedDocument(
            doc_id=doc_id,
            title=metadata.get("title", "未命名文档"),
            modality="text",
            chunks=[chunk],
            raw_metadata=metadata,
        )


class ImageParser:
    """图片解析器——提取图片描述文本"""

    def __init__(self, llm_client=None, llm_model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def parse(self, image_data: Union[str, bytes], doc_id: str, metadata: Optional[dict] = None) -> ParsedDocument:
        metadata = metadata or {}

        # 如果是 base64 字符串，解码
        if isinstance(image_data, str):
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception:
                # 可能是文件路径
                image_bytes = Path(image_data).read_bytes()
        else:
            image_bytes = image_data

        # 生成图片描述
        description = self._describe_image(image_bytes, metadata)

        chunk = DocumentChunk(
            chunk_id=f"{doc_id}_chunk_0000",
            doc_id=doc_id,
            content=description,
            modality="image",
            metadata={**metadata, "image_size": len(image_bytes)},
        )
        return ParsedDocument(
            doc_id=doc_id,
            title=metadata.get("title", "未命名图片"),
            modality="image",
            chunks=[chunk],
            raw_metadata=metadata,
        )

    def _describe_image(self, image_bytes: bytes, metadata: dict) -> str:
        """使用 LLM 生成图片描述"""
        if self.llm_client:
            try:
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "请详细描述这张图片的内容，包括文字、物体、场景等。"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            ],
                        }
                    ],
                    max_tokens=500,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM 图片描述失败: {e}，使用基础描述")

        # 降级：使用 OCR
        try:
            import pytesseract
            from PIL import Image
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")
            if text.strip():
                return f"[图片OCR内容]: {text.strip()}"
        except Exception as e:
            logger.warning(f"OCR 失败: {e}")

        return f"[图片: {metadata.get('title', '未命名')}]"


class TableParser:
    """表格解析器"""

    def parse(self, content: Union[str, bytes], doc_id: str, metadata: Optional[dict] = None) -> ParsedDocument:
        metadata = metadata or {}

        # 尝试多种解析方式
        table_text = self._parse_table(content)

        chunk = DocumentChunk(
            chunk_id=f"{doc_id}_chunk_0000",
            doc_id=doc_id,
            content=table_text,
            modality="table",
            metadata=metadata,
        )
        return ParsedDocument(
            doc_id=doc_id,
            title=metadata.get("title", "未命名表格"),
            modality="table",
            chunks=[chunk],
            raw_metadata=metadata,
        )

    def _parse_table(self, content: Union[str, bytes]) -> str:
        """解析表格为文本表示"""
        # 尝试 Camelot
        try:
            import camelot
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                if isinstance(content, str):
                    f.write(content.encode())
                else:
                    f.write(content)
                f.flush()
                tables = camelot.read_pdf(f.name, pages="1")
                if len(tables) > 0:
                    return tables[0].df.to_string(index=False)
        except Exception:
            pass

        # 尝试 Tabula
        try:
            import tabula
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                if isinstance(content, str):
                    f.write(content.encode())
                else:
                    f.write(content)
                f.flush()
                dfs = tabula.read_pdf(f.name, pages="1")
                if dfs:
                    return dfs[0].to_string(index=False)
        except Exception:
            pass

        # 如果是 CSV/TSV 文本
        if isinstance(content, str):
            lines = content.strip().split("\n")
            if any("\t" in line or "," in line for line in lines):
                return content

        return f"[表格: 无法解析]"


class PDFParser:
    """PDF 解析器——混合内容解析"""

    def __init__(self, llm_client=None, llm_model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.text_parser = TextParser()
        self.table_parser = TableParser()
        self.image_parser = ImageParser(llm_client, llm_model)

    def parse(self, file_path: Union[str, Path], doc_id: str, metadata: Optional[dict] = None) -> ParsedDocument:
        metadata = metadata or {}
        file_path = Path(file_path)

        try:
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(
                filename=str(file_path),
                strategy="hi_res",
                extract_images_in_pdf=True,
                infer_table_structure=True,
            )
        except Exception as e:
            logger.warning(f"Unstructured PDF 解析失败: {e}，降级为文本提取")
            return self._fallback_parse(file_path, doc_id, metadata)

        chunks = []
        for i, element in enumerate(elements):
            modality = "text"
            if "Table" in type(element).__name__:
                modality = "table"
            elif "Image" in type(element).__name__:
                modality = "image"

            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{i:04d}",
                doc_id=doc_id,
                content=str(element),
                modality=modality,
                metadata={**metadata, "element_type": type(element).__name__},
            )
            chunks.append(chunk)

        return ParsedDocument(
            doc_id=doc_id,
            title=metadata.get("title", file_path.name),
            modality="mixed",
            chunks=chunks,
            raw_metadata=metadata,
        )

    def _fallback_parse(self, file_path: Path, doc_id: str, metadata: dict) -> ParsedDocument:
        """降级解析：使用 PyMuPDF 提取文本"""
        try:
            import fitz
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return self.text_parser.parse(text, doc_id, metadata)
        except Exception as e:
            logger.error(f"PDF 降级解析也失败: {e}")
            return ParsedDocument(
                doc_id=doc_id,
                title=metadata.get("title", file_path.name),
                modality="text",
                chunks=[DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_0000",
                    doc_id=doc_id,
                    content=f"[无法解析PDF: {file_path.name}]",
                    modality="text",
                    metadata=metadata,
                )],
                raw_metadata=metadata,
            )


class MultiModalParser:
    """多模态解析器——统一入口"""

    def __init__(self, llm_client=None, llm_model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.parsers = {
            "text": TextParser(),
            "image": ImageParser(llm_client, llm_model),
            "table": TableParser(),
            "pdf": PDFParser(llm_client, llm_model),
        }

    def parse(
        self,
        content: Union[str, bytes],
        modality: str,
        doc_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ParsedDocument:
        """解析多模态内容"""
        import uuid
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        metadata = metadata or {}

        parser = self.parsers.get(modality)
        if not parser:
            logger.warning(f"不支持的模态: {modality}，降级为文本解析")
            parser = self.parsers["text"]

        return parser.parse(content, doc_id, metadata)

    def parse_file(self, file_path: Union[str, Path], metadata: Optional[dict] = None) -> ParsedDocument:
        """解析文件（自动检测类型）"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        import uuid
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        metadata = metadata or {"title": file_path.name, "source": str(file_path)}

        if suffix == ".pdf":
            return self.parsers["pdf"].parse(file_path, doc_id, metadata)
        elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".bmp"):
            content = file_path.read_bytes()
            return self.parsers["image"].parse(content, doc_id, metadata)
        elif suffix in (".csv", ".tsv", ".xlsx", ".xls"):
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            return self.parsers["table"].parse(content, doc_id, metadata)
        else:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            return self.parsers["text"].parse(content, doc_id, metadata)