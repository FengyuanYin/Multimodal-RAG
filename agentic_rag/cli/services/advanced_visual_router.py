"""Post-rerank VLM routing for image evidence in Advanced RAG."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ..errors import CancelledError


class AdvancedVisualRouter:
    """Classify and reconstruct only images referenced by final Advanced hits."""

    PROMPT_VERSION = "advanced-visual-router-v1"
    MAX_IMAGE_BYTES = 10 * 1024 * 1024
    PRIMARY_PROMPT = """Analyze only visible content in this document image.
Return exactly one JSON object with keys:
- image_type: data_chart, structure_diagram, or other
- markdown_table: for data_chart, a Markdown table covering visible title, axes/series/values and uncertain cells; otherwise empty
- mermaid: for structure_diagram, Mermaid text reconstructing visible nodes, labels, directions and edges; otherwise empty
- description: for other, a factual visual description; for typed images it may add concise context
- uncertainty: unreadable or ambiguous text, values, directions, or relationships
Do not invent hidden data or relationships. Do not wrap the JSON in prose."""
    FALLBACK_PROMPT = """Describe this document image using only visible facts. Include visible text, layout or data, key conclusions, and uncertainty. Do not invent unreadable values or hidden relationships."""

    def __init__(self, knowledge, vlm_client) -> None:
        self.knowledge = knowledge
        self.vlm_client = vlm_client

    @staticmethod
    def _report() -> dict[str, Any]:
        return {
            "eligible_images": 0,
            "unique_images": 0,
            "cache_hits": 0,
            "primary_calls": 0,
            "fallback_calls": 0,
            "analyzed": 0,
            "degraded": [],
        }

    @staticmethod
    def _hit_is_eligible(hit) -> bool:
        scores = getattr(hit, "channel_scores", {}) or {}
        return "multimodal" in scores or any(str(name).endswith("reference_graph") for name in scores)

    def _resolve_images(self, hits, report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
        media_by_document: dict[str, dict[str, dict[str, Any]]] = {}
        references: dict[str, list[Any]] = {}
        ordered_ids: list[str] = []
        missing: set[str] = set()
        for hit in hits:
            if not self._hit_is_eligible(hit):
                continue
            for ref in getattr(hit, "media_refs", []) or []:
                media_id = str(ref.get("media_id") or "").strip()
                if not media_id:
                    continue
                if str(ref.get("media_type") or ref.get("type") or "image") != "image":
                    continue
                report["eligible_images"] += 1
                if media_id not in references:
                    references[media_id] = []
                    ordered_ids.append(media_id)
                if hit not in references[media_id]:
                    references[media_id].append(hit)
                if hit.document_id not in media_by_document:
                    media_by_document[hit.document_id] = {
                        str(item["id"]): item for item in self.knowledge.list_media(hit.document_id)
                    }
                media = media_by_document[hit.document_id].get(media_id)
                if media is None:
                    missing.add(media_id)
        resolved: list[dict[str, Any]] = []
        for media_id in ordered_ids:
            media = None
            for hit in references[media_id]:
                media = media_by_document.get(hit.document_id, {}).get(media_id)
                if media is not None:
                    break
            if media is None:
                report["degraded"].append({"media_id": media_id, "reason": "media_not_found"})
                continue
            if str(media.get("media_type")) != "image":
                references.pop(media_id, None)
                continue
            resolved.append(media)
        report["unique_images"] = len(resolved)
        return resolved, references

    @classmethod
    def _read_image(cls, media: dict[str, Any]) -> tuple[bytes, str]:
        configured = Path(str(media.get("storage_path") or ""))
        if configured.is_symlink():
            raise ValueError("unsafe_symlink")
        path = configured.resolve()
        if not path.is_file():
            raise ValueError("image_missing")
        size = path.stat().st_size
        if size > cls.MAX_IMAGE_BYTES:
            raise ValueError("image_too_large")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != str(media.get("checksum") or ""):
            raise ValueError("checksum_mismatch")
        return raw, str(media.get("mime_type") or "application/octet-stream")

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        value = str(text or "").strip()
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.IGNORECASE | re.DOTALL)
        if fence:
            value = fence.group(1).strip()
        else:
            start, end = value.find("{"), value.rfind("}")
            if start >= 0 and end > start:
                value = value[start:end + 1]
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid_json") from exc
        if not isinstance(parsed, dict):
            raise ValueError("invalid_json_object")
        return parsed

    @staticmethod
    def _valid_markdown_table(value: str) -> bool:
        lines = [line.strip() for line in str(value or "").strip().splitlines() if line.strip()]
        if len(lines) < 3 or not all("|" in line for line in lines[:3]):
            return False
        cells = [cell.strip() for cell in lines[1].strip("|").split("|")]
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)

    @staticmethod
    def _normalize_mermaid(value: str) -> str:
        content = str(value or "").strip()
        fence = re.fullmatch(r"```(?:mermaid)?\s*(.*?)\s*```", content, flags=re.IGNORECASE | re.DOTALL)
        return fence.group(1).strip() if fence else content

    @classmethod
    def _valid_mermaid(cls, value: str) -> bool:
        content = cls._normalize_mermaid(value)
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        if not re.match(r"^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|erDiagram)\b", lines[0], re.IGNORECASE):
            return False
        return any(token in "\n".join(lines[1:]) for token in ("-->", "---", "->>", "-->>", ":", "{"))

    @classmethod
    def _parse_primary(cls, text: str, media_id: str) -> dict[str, Any]:
        parsed = cls._extract_json(text)
        image_type = str(parsed.get("image_type") or "").strip()
        uncertainty = str(parsed.get("uncertainty") or "").strip()
        if image_type == "data_chart":
            content = str(parsed.get("markdown_table") or "").strip()
            if not cls._valid_markdown_table(content):
                raise ValueError("invalid_markdown_table")
            representation = "markdown_table"
        elif image_type == "structure_diagram":
            content = cls._normalize_mermaid(str(parsed.get("mermaid") or ""))
            if not cls._valid_mermaid(content):
                raise ValueError("invalid_mermaid")
            representation = "mermaid"
        elif image_type == "other":
            content = str(parsed.get("description") or "").strip()
            if not content:
                raise ValueError("invalid_description")
            representation = "narrative"
        else:
            raise ValueError("invalid_image_type")
        return {
            "media_id": media_id,
            "image_type": image_type,
            "representation": representation,
            "content": content,
            "fallback_used": False,
            "uncertainty": uncertainty,
            "cached": False,
        }

    @staticmethod
    def _fallback_evidence(text: str, media_id: str) -> dict[str, Any]:
        content = str(text or "").strip()
        if not content:
            raise ValueError("invalid_fallback_description")
        return {
            "media_id": media_id,
            "image_type": "fallback",
            "representation": "narrative",
            "content": content,
            "fallback_used": True,
            "uncertainty": "Typed reconstruction failed; general visual understanding used.",
            "cached": False,
        }

    @staticmethod
    def _attach(evidence: dict[str, Any], hits: list[Any]) -> None:
        for hit in hits:
            if not any(item.get("media_id") == evidence["media_id"] for item in hit.visual_evidence):
                hit.visual_evidence.append(dict(evidence))

    @staticmethod
    def _reason(exc: Exception) -> str:
        if isinstance(exc, ValueError) and str(exc):
            return str(exc)
        return f"vlm_error:{type(exc).__name__}"

    def enrich(self, hits, cancel) -> dict[str, Any]:
        report = self._report()
        media_items, references = self._resolve_images(hits, report)
        if not media_items:
            return report
        if self.vlm_client is None:
            report["degraded"].extend(
                {"media_id": str(media["id"]), "reason": "vlm_not_configured"} for media in media_items
            )
            return report

        profile = str(self.vlm_client.profile_fingerprint)
        for media in media_items:
            cancel.checkpoint()
            media_id = str(media["id"])
            checksum = str(media.get("checksum") or "")
            cached = self.knowledge.get_media_vlm_analysis(
                media_id, checksum, profile, self.PROMPT_VERSION
            )
            if cached:
                evidence = dict(cached)
                evidence["cached"] = True
                self._attach(evidence, references[media_id])
                report["cache_hits"] += 1
                report["analyzed"] += 1
                continue
            try:
                raw, mime_type = self._read_image(media)
                report["primary_calls"] += 1
                primary = self.vlm_client.describe_image(raw, mime_type, self.PRIMARY_PROMPT, cancel)
                try:
                    evidence = self._parse_primary(primary, media_id)
                except ValueError:
                    report["fallback_calls"] += 1
                    fallback = self.vlm_client.describe_image(raw, mime_type, self.FALLBACK_PROMPT, cancel)
                    evidence = self._fallback_evidence(fallback, media_id)
                self.knowledge.upsert_media_vlm_analysis(
                    media_id, checksum, profile, self.PROMPT_VERSION, evidence
                )
                self._attach(evidence, references[media_id])
                report["analyzed"] += 1
            except CancelledError:
                raise
            except Exception as exc:
                report["degraded"].append({"media_id": media_id, "reason": self._reason(exc)})
        return report
