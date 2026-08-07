"""
多模态解析器模块
==============
支持文本、图片、表格、PDF 等多模态文档的解析与统一表示。
"""

from typing import List, Optional, Union, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import base64
import hashlib
import io
import mimetypes
import re
from loguru import logger


@dataclass
class MediaRef:
    """媒体引用：文本块中对图片/表格的引用位置"""
    media_id: str
    media_type: str  # image | table
    label: str       # 引用的标签文本，如 "图1" / "Table 2"
    page: int = 1    # 媒体所在页码
    offset: int = 0  # 引用在文本中的起始偏移
    confidence: float = 0.0
    resolution: str = "unresolved"
    reason: str = ""


@dataclass
class MediaAsset:
    """多媒体资产（图片/表格），由文本块通过 MediaRef 引用"""
    id: str
    doc_id: str
    type: str            # image | table
    page: int = 1
    label: str = ""      # 原始标签，如 "图1"
    caption: str = ""    # 说明文字（图片描述 / 表格文本表示）
    data: Optional[str] = None  # 图片 base64 或表格原始文本（可选，便于 VLM 使用）
    search_text: str = ""
    mime_type: str = ""
    checksum: str = ""
    extraction_method: str = ""
    quality: str = "derived"  # exact | derived | fallback
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """统一文档块"""
    chunk_id: str
    doc_id: str
    content: str
    modality: str = "text"  # text | image | table | mixed
    metadata: dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    media_refs: List[MediaRef] = field(default_factory=list)  # 引用位置记录


@dataclass
class ParsedDocument:
    """解析后的文档"""
    doc_id: str
    title: str
    modality: str
    chunks: List[DocumentChunk]
    raw_metadata: dict = field(default_factory=dict)
    media: List[MediaAsset] = field(default_factory=list)  # 抽取出的媒体资产


# ── 引用位置检测（RAG-Anything 风格：识别文本中对图/表的引用） ──

# 匹配 "图1" / "图 1" / "Figure 1" / "Fig. 1" / "表2" / "表 2" / "Table 2"
_MEDIA_REF_PATTERNS = [
    re.compile(r"(?:图\s*|Figure\s+|Fig\.?\s*)(\d{1,3})", re.IGNORECASE),
    re.compile(r"(?:表\s*|表格\s*|Table\s+)(\d{1,3})", re.IGNORECASE),
]


def detect_media_refs(text: str, doc_id: str, page: int = 1,
                      media_index: Optional[Dict[str, MediaAsset]] = None) -> List[MediaRef]:
    """
    检测文本中对图片/表格的引用位置。

    Args:
        text: 文本块内容
        doc_id: 文档 ID
        page: 页码
        media_index: {label_key: MediaAsset} 映射（如 {"图1": asset}），用于把标签关联到资产

    Returns:
        List[MediaRef]: 引用位置列表
    """
    from agentic_rag.memory.media_association import detect_references, resolve_reference

    assets = list((media_index or {}).values())
    unique_assets = list({asset.id: asset for asset in assets}.values())
    refs = []
    for detected in detect_references(text, doc_id, page):
        if not unique_assets:
            media_id = f"{doc_id}_{detected.media_type}_{detected.label[1:]}"
            confidence, resolution, reason = 0.0, "unresolved", "尚未提供媒体资产索引"
        else:
            decision = resolve_reference(detected, unique_assets)
            media_id = decision.media_id
            confidence, resolution, reason = decision.confidence, decision.resolution, decision.reason
        refs.append(MediaRef(
            media_id=media_id,
            media_type=detected.media_type,
            label=detected.label,
            page=detected.page,
            offset=detected.offset,
            confidence=confidence,
            resolution=resolution,
            reason=reason,
        ))
    return refs


def _ref_to_dict(ref: MediaRef) -> dict:
    """MediaRef -> dict（用于写入向量/BM25 索引与 API 响应）"""
    return {
        "media_id": ref.media_id,
        "media_type": ref.media_type,
        "label": ref.label,
        "page": ref.page,
        "offset": ref.offset,
        "confidence": ref.confidence,
        "resolution": ref.resolution,
        "reason": ref.reason,
    }


def refs_to_dicts(refs: List[MediaRef]) -> List[dict]:
    """批量转换引用位置为 dict"""
    return [_ref_to_dict(r) for r in refs]


def _media_to_dict(media: MediaAsset, include_data: bool = False) -> dict:
    """MediaAsset -> dict（用于 API 响应）"""
    d = {
        "id": media.id,
        "doc_id": media.doc_id,
        "type": media.type,
        "page": media.page,
        "label": media.label,
        "caption": media.caption,
        "search_text": media.search_text,
        "mime_type": media.mime_type,
        "checksum": media.checksum,
        "extraction_method": media.extraction_method,
        "quality": media.quality,
        "metadata": media.metadata,
    }
    if include_data and media.data:
        d["data"] = media.data
    return d


class TextParser:
    """文本解析器"""

    def parse(self, content: str, doc_id: str, metadata: Optional[dict] = None) -> ParsedDocument:
        metadata = metadata or {}
        refs = detect_media_refs(content, doc_id, page=1)
        chunk = DocumentChunk(
            chunk_id=f"{doc_id}_chunk_0000",
            doc_id=doc_id,
            content=content,
            modality="text",
            metadata={**metadata, "media_refs": refs_to_dicts(refs)},
            media_refs=refs,
        )
        return ParsedDocument(
            doc_id=doc_id,
            title=metadata.get("title", "未命名文档"),
            modality="text",
            chunks=[chunk],
            raw_metadata=metadata,
            media=[],
        )


class ImageParser:
    """图片解析器——提取图片描述文本（优先使用 VLM，其次 LLM，最后 OCR）"""

    def __init__(self, llm_client=None, llm_model: str = "gpt-4o",
                 vlm_client=None, vlm_model: Optional[str] = None):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.vlm_client = vlm_client
        self.vlm_model = vlm_model

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
        checksum = hashlib.sha256(image_bytes).hexdigest()
        mime_type = metadata.get("mime_type") or mimetypes.guess_type(str(metadata.get("source", "")))[0] or "image/png"
        media_id = f"{doc_id}_image_{checksum[:12]}"
        asset = MediaAsset(
            id=media_id,
            doc_id=doc_id,
            type="image",
            page=int(metadata.get("page", 1) or 1),
            label=metadata.get("label", "图1"),
            caption=description,
            data=base64.b64encode(image_bytes).decode("utf-8"),
            search_text=description,
            mime_type=mime_type,
            checksum=checksum,
            extraction_method="image_parser",
            quality="derived",
            metadata=metadata,
        )
        ref = MediaRef(
            media_id=media_id,
            media_type="image",
            label=asset.label,
            page=asset.page,
            offset=0,
            confidence=1.0,
            resolution="exact",
            reason="独立图片与描述块直接关联",
        )

        chunk = DocumentChunk(
            chunk_id=f"{doc_id}_chunk_0000",
            doc_id=doc_id,
            content=description,
            modality="image",
            metadata={**metadata, "image_size": len(image_bytes), "media_refs": refs_to_dicts([ref])},
            media_refs=[ref],
        )
        return ParsedDocument(
            doc_id=doc_id,
            title=metadata.get("title", "未命名图片"),
            modality="image",
            chunks=[chunk],
            raw_metadata=metadata,
            media=[asset],
        )

    def _describe_image(self, image_bytes: bytes, metadata: dict) -> str:
        """生成图片描述：VLM > LLM > OCR > 占位"""
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        # 1. 优先使用 VLM（视觉语言模型）
        if self.vlm_client:
            try:
                response = self.vlm_client.chat.completions.create(
                    model=self.vlm_model or self.llm_model,
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
                logger.warning(f"VLM 图片描述失败: {e}")

        # 2. 使用 LLM（部分模型也支持图像输入）
        if self.llm_client:
            try:
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

        # 3. 降级：使用 OCR
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
        checksum = hashlib.sha256(table_text.encode("utf-8")).hexdigest()
        page = int(metadata.get("page", 1) or 1)
        label = metadata.get("label", "表1")
        asset = MediaAsset(
            id=f"{doc_id}_table_{checksum[:12]}",
            doc_id=doc_id,
            type="table",
            page=page,
            label=label,
            caption=table_text[:1000],
            data=table_text,
            search_text=table_text,
            mime_type="text/plain",
            checksum=checksum,
            extraction_method="table_parser",
            quality="derived",
            metadata=metadata,
        )
        ref = MediaRef(asset.id, "table", label, page, 0, 1.0, "exact", "独立表格与文本块直接关联")
        chunk.media_refs = [ref]
        chunk.metadata = {**metadata, "media_refs": refs_to_dicts([ref])}
        return ParsedDocument(
            doc_id=doc_id,
            title=metadata.get("title", "未命名表格"),
            modality="table",
            chunks=[chunk],
            raw_metadata=metadata,
            media=[asset],
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
    """PDF 解析器——混合内容解析（文本/表格/图片 + 图/表引用位置）"""

    def __init__(self, llm_client=None, llm_model: str = "gpt-4o",
                 vlm_client=None, vlm_model: Optional[str] = None):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.vlm_client = vlm_client
        self.vlm_model = vlm_model
        self.text_parser = TextParser()
        self.table_parser = TableParser()
        self.image_parser = ImageParser(llm_client, llm_model, vlm_client, vlm_model)

    def parse(self, file_path: Union[str, Path], doc_id: str, metadata: Optional[dict] = None) -> ParsedDocument:
        metadata = metadata or {}
        file_path = Path(file_path)

        # 1. 抽取页内图片（PyMuPDF），用于构建图片资产与引用
        media, page_images = self._extract_media(file_path, doc_id)

        try:
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(
                filename=str(file_path),
                strategy="hi_res",
                extract_images_in_pdf=True,
                infer_table_structure=True,
            )
        except Exception as e:
            logger.warning(
                f"Unstructured PDF 解析失败: {e}，降级为文本提取。"
                "如需完整版面/表格/图片解析，请安装: pip install agentic-rag[pdf]"
            )
            return self._fallback_parse(file_path, doc_id, metadata, media, page_images)

        chunks = []
        media_index = {m.label: m for m in media if m.label}
        img_seq = len([m for m in media if m.type == "image"])
        table_seq = len([m for m in media if m.type == "table"])
        for i, element in enumerate(elements):
            modality = "text"
            if "Table" in type(element).__name__:
                modality = "table"
            elif "Image" in type(element).__name__:
                modality = "image"

            content = str(element)
            el_meta = getattr(element, "metadata", None)
            page_number = int(getattr(el_meta, "page_number", 1) or 1)
            # 检测文本中对图/表的引用位置（RAG-Anything 风格）
            refs = detect_media_refs(content, doc_id, page=page_number, media_index=media_index)

            if modality == "table":
                table_seq += 1
                label = f"表{table_seq}"
                table_text = content.strip() or "[表格: 无法解析]"
                checksum = hashlib.sha256(table_text.encode("utf-8")).hexdigest()
                asset = MediaAsset(
                    id=f"{doc_id}_table_{checksum[:12]}", doc_id=doc_id, type="table",
                    page=page_number, label=label, caption=table_text[:1000], data=table_text,
                    search_text=table_text, mime_type="text/plain", checksum=checksum,
                    extraction_method="unstructured", quality="exact",
                    metadata={"source": str(file_path), "element_type": type(element).__name__},
                )
                media.append(asset)
                media_index[label] = asset
                refs.append(MediaRef(asset.id, "table", label, page_number, 0, 1.0, "exact", "表格元素直接关联"))

            if modality == "image":
                # 关联/注册图片资产：优先使用 Unstructured 元素自带的 base64，
                # 否则尽力关联 PyMuPDF 已抽取的页内图片
                img_b64 = None
                try:
                    img_b64 = getattr(el_meta, "image_base64", None) if el_meta else None
                except Exception:
                    img_b64 = None
                if img_b64 and not any(m.data == img_b64 for m in media):
                    img_seq += 1
                    asset = MediaAsset(
                        id=f"{doc_id}_image_{img_seq}",
                        doc_id=doc_id,
                        type="image",
                        page=page_number,
                        label=f"图{img_seq}",
                        caption=content.strip(),
                        data=img_b64,
                        search_text=content.strip(),
                        mime_type="image/png",
                        checksum=hashlib.sha256(base64.b64decode(img_b64)).hexdigest(),
                        extraction_method="unstructured",
                        quality="exact",
                        metadata={"source": str(file_path), "element_type": type(element).__name__},
                    )
                    media.append(asset)
                    media_index[asset.label] = asset
                    refs.append(MediaRef(
                        media_id=asset.id, media_type="image", label=asset.label,
                        page=page_number, offset=0, confidence=1.0,
                        resolution="exact", reason="图片元素直接关联",
                    ))
                elif not refs and media:
                    # 无 base64 时关联最后一张已抽取图片（尽力而为）
                    page_imgs = [m for m in media if m.type == "image" and m.page == page_number]
                    if page_imgs:
                        target = page_imgs[-1]
                        refs.append(MediaRef(
                            media_id=target.id, media_type="image", label=target.label,
                            page=page_number, offset=0, confidence=0.4,
                            resolution="page_match", reason="图片元素无数据，仅按同页候选降级关联",
                        ))
                if not content.strip():
                    content = f"[图片: {refs[0].label if refs else '未命名'}]"

            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{i:04d}",
                doc_id=doc_id,
                content=content,
                modality=modality,
                metadata={
                    **metadata,
                    "element_type": type(element).__name__,
                    "page": page_number,
                    "media_refs": [_ref_to_dict(r) for r in refs],
                },
                media_refs=refs,
            )
            chunks.append(chunk)

        # 2. 若未能从 Unstructured 拿到元素，回退到页级文本 + 页图
        if not chunks:
            return self._fallback_parse(file_path, doc_id, metadata, media, page_images)

        return ParsedDocument(
            doc_id=doc_id,
            title=metadata.get("title", file_path.name),
            modality="mixed",
            chunks=chunks,
            raw_metadata=metadata,
            media=media,
        )

    def _extract_media(self, file_path: Path, doc_id: str) -> Tuple[List[MediaAsset], Dict[int, str]]:
        """使用 PyMuPDF 抽取页内图片与页级图像（base64）。

        Returns:
            (media_assets, page_images): 媒体资产列表；{page_no: 页面渲染 base64}
        """
        media: List[MediaAsset] = []
        page_images: Dict[int, str] = {}
        try:
            import fitz
        except Exception as e:
            logger.warning(f"PyMuPDF 不可用，跳过图片抽取: {e}")
            return media, page_images

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.warning(f"PyMuPDF 打开失败: {e}")
            return media, page_images

        img_seq = 0
        try:
            for page_no, page in enumerate(doc, start=1):
                # 页级缩略图（供展示/引用兜底）
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
                    page_images[page_no] = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                except Exception:
                    pass

                # 页内嵌入图片
                try:
                    for img in page.get_images(full=True):
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n - pix.alpha > 3:  # CMYK 转 RGB
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_bytes = pix.tobytes("png")
                        img_seq += 1
                        media_id = f"{doc_id}_image_{img_seq}"
                        media.append(MediaAsset(
                            id=media_id,
                            doc_id=doc_id,
                            type="image",
                            page=page_no,
                            label=f"图{img_seq}",
                            caption="",
                            data=base64.b64encode(img_bytes).decode("utf-8"),
                            search_text="",
                            mime_type="image/png",
                            checksum=hashlib.sha256(img_bytes).hexdigest(),
                            extraction_method="pymupdf",
                            quality="exact",
                            metadata={"source": str(file_path), "xref": xref},
                        ))
                except Exception as e:
                    logger.debug(f"页 {page_no} 图片抽取失败: {e}")
        except Exception as e:
            logger.warning(f"PDF 图片抽取中断: {e}")
        finally:
            doc.close()

        return media, page_images

    def _fallback_parse(self, file_path: Path, doc_id: str, metadata: dict,
                        media: Optional[List[MediaAsset]] = None,
                        page_images: Optional[Dict[int, str]] = None) -> ParsedDocument:
        """降级解析：使用 PyMuPDF 提取文本 + 检测引用位置"""
        media = media or []
        page_images = page_images or {}
        try:
            import fitz
            doc = fitz.open(file_path)
            chunks = []
            media_index = {m.label: m for m in media if m.label}
            for page_no, page in enumerate(doc, start=1):
                text = page.get_text()
                if not text.strip():
                    continue
                refs = detect_media_refs(text, doc_id, page=page_no, media_index=media_index)
                # 每页作为独立分块，便于按页码记录引用位置
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{page_no:04d}",
                    doc_id=doc_id,
                    content=text,
                    modality="text",
                    metadata={**metadata, "page": page_no, "media_refs": [_ref_to_dict(r) for r in refs]},
                    media_refs=refs,
                ))
            doc.close()
            if chunks:
                return ParsedDocument(
                    doc_id=doc_id,
                    title=metadata.get("title", file_path.name),
                    modality="mixed",
                    chunks=chunks,
                    raw_metadata=metadata,
                    media=media,
                )
            # 无文本 → 用页图作为媒体
            for page_no, img_b64 in page_images.items():
                media.append(MediaAsset(
                    id=f"{doc_id}_page_{page_no}",
                    doc_id=doc_id,
                    type="image",
                    page=page_no,
                    label=f"第{page_no}页",
                    caption=f"第{page_no}页图像",
                    data=img_b64,
                    search_text=f"第{page_no}页页面快照",
                    mime_type="image/png",
                    checksum=hashlib.sha256(base64.b64decode(img_b64)).hexdigest(),
                    extraction_method="pymupdf_page_render",
                    quality="fallback",
                    metadata={"source": str(file_path), "page_image": True},
                ))
            return ParsedDocument(
                doc_id=doc_id,
                title=metadata.get("title", file_path.name),
                modality="text",
                chunks=[DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_0000",
                    doc_id=doc_id,
                    content=f"[无法提取文本，已保存 {len(media)} 张页面图像]",
                    modality="text",
                    metadata=metadata,
                )],
                raw_metadata=metadata,
                media=media,
            )
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
                media=media,
            )


class MultiModalParser:
    """多模态解析器——统一入口（文本/图片/表格/PDF）"""

    def __init__(self, llm_client=None, llm_model: str = "gpt-4o",
                 vlm_client=None, vlm_model: Optional[str] = None):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.vlm_client = vlm_client
        self.vlm_model = vlm_model
        self.parsers = {
            "text": TextParser(),
            "image": ImageParser(llm_client, llm_model, vlm_client, vlm_model),
            "table": TableParser(),
            "pdf": PDFParser(llm_client, llm_model, vlm_client, vlm_model),
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
