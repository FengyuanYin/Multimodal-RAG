"""确定性的文本引用与媒体关联规则。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, List, Mapping, Optional
import re


_PATTERN = re.compile(
    r"(?P<image>(?:图\s*|Figure\s+|Fig\.?\s*)(?P<image_no>\d{1,3}))"
    r"|(?P<table>(?:表格?\s*|Table\s+)(?P<table_no>\d{1,3}))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DetectedReference:
    document_id: str
    media_type: str
    label: str
    page: int
    offset: int
    raw_label: str


@dataclass(frozen=True)
class AssociationDecision:
    media_id: str
    confidence: float
    resolution: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def detect_references(text: str, document_id: str, page: int = 1) -> List[DetectedReference]:
    refs: List[DetectedReference] = []
    for match in _PATTERN.finditer(text or ""):
        if match.group("image_no"):
            media_type, number = "image", match.group("image_no")
        else:
            media_type, number = "table", match.group("table_no")
        refs.append(DetectedReference(
            document_id=document_id,
            media_type=media_type,
            label=f"{'图' if media_type == 'image' else '表'}{number}",
            page=max(1, int(page or 1)),
            offset=match.start(),
            raw_label=match.group(0),
        ))
    return refs


def _value(asset: Any, key: str, default: Any = None) -> Any:
    if isinstance(asset, Mapping):
        return asset.get(key, default)
    return getattr(asset, key, default)


def resolve_reference(reference: DetectedReference, media: Iterable[Any]) -> AssociationDecision:
    document_assets = [
        item for item in media
        if _value(item, "doc_id", _value(item, "document_id", "")) == reference.document_id
    ]
    candidates = [item for item in document_assets if _value(item, "type", "") == reference.media_type]
    same_label = [item for item in candidates if _value(item, "label", "") == reference.label]
    same_page_label = [item for item in same_label if int(_value(item, "page", 1) or 1) == reference.page]

    if len(same_page_label) == 1:
        return AssociationDecision(str(_value(same_page_label[0], "id", "")), 1.0, "exact", "同文档、同页、同类型、同标签唯一匹配")
    if len(same_label) == 1:
        return AssociationDecision(str(_value(same_label[0], "id", "")), 0.85, "unique_label", "同文档内同类型标签唯一匹配")

    parser_linked = [
        item for item in candidates
        if _value(item, "metadata", {}).get("reference_label") == reference.label
        and int(_value(item, "page", 1) or 1) == reference.page
    ]
    if len(parser_linked) == 1:
        return AssociationDecision(str(_value(parser_linked[0], "id", "")), 0.9, "layout", "解析器提供明确版面关系")

    page_assets = [item for item in candidates if int(_value(item, "page", 1) or 1) == reference.page]
    page_snapshots = [
        item for item in document_assets
        if _value(item, "type", "") == "page_snapshot"
        and int(_value(item, "page", 1) or 1) == reference.page
    ]
    if len(page_snapshots) == 1:
        return AssociationDecision(str(_value(page_snapshots[0], "id", "")), 0.35, "page_match", "仅定位到低置信度页面快照")

    reason = "存在多个候选，无法确定" if same_label or page_assets else "未找到对应媒体资产"
    return AssociationDecision("", 0.0, "unresolved", reason)


def associate_references(text: str, document_id: str, page: int, media: Iterable[Any]) -> List[dict]:
    output = []
    for ref in detect_references(text, document_id, page):
        decision = resolve_reference(ref, media)
        output.append({
            "media_id": decision.media_id,
            "media_type": ref.media_type,
            "label": ref.label,
            "page": ref.page,
            "offset": ref.offset,
            "confidence": decision.confidence,
            "resolution": decision.resolution,
            "reason": decision.reason,
        })
    return output
