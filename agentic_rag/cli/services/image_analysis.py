"""Authorized on-demand VLM analysis for document-bound images."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..errors import ConfigurationError, UsageError
from ..models import EventKind, OutputEvent


class ImageAnalysisService:
    PROMPT_VERSION = "image-table-v1"

    def __init__(self, knowledge, repository, files, vlm_client) -> None:
        self.knowledge, self.repository, self.files, self.vlm_client = knowledge, repository, files, vlm_client

    def analyze(self, workspace: dict, media_id: str, purpose: str, output, cancel) -> str:
        media = next((item for item in self.knowledge.list_media(workspace["document_id"]) if item["id"] == media_id), None)
        if not media: raise UsageError("Image is not attached to the active document")
        if not self.vlm_client: raise ConfigurationError("VLM is not configured; the main LLM can continue from Markdown text")
        path = Path(media["storage_path"]).resolve()
        if not path.is_file() or path.is_symlink(): raise ConfigurationError("Document image is missing or unsafe")
        raw = path.read_bytes()
        if len(raw) > 10 * 1024 * 1024: raise ConfigurationError("Image exceeds the 10MB VLM request limit")
        if hashlib.sha256(raw).hexdigest() != media["checksum"]: raise ConfigurationError("Document image integrity check failed")
        canonical = " ".join(purpose.lower().split())[:500]
        key = hashlib.sha256(f'{workspace["id"]}\0{media["checksum"]}\0{canonical}\0{self.vlm_client.profile_fingerprint}\0{self.PROMPT_VERSION}'.encode()).hexdigest()
        cached = self.repository.cache_get(key)
        if cached: return cached["result_text"]
        output.emit(OutputEvent(EventKind.PROGRESS, phase="image_analysis", text=f"Analyzing image {media_id}"))
        prompt = f"Analyze this document image for: {canonical}. Return a Markdown table with rows: Image type, Title/number, Visible text, Structure/data, Key conclusions, Uncertainty. State only visible facts."
        result = self.vlm_client.describe_image(raw, media["mime_type"], prompt, cancel).strip()
        if "|" not in result:
            result = "| Field | Content |\n|---|---|\n| VLM analysis | " + result.replace("|", "\\|").replace("\n", "<br>") + " |"
        record = self.files.create_markdown(workspace["id"], "image_analysis", f"{media_id}-analysis.md", result, canonical)
        self.repository.cache_put(key, workspace["id"], "read_image", result, record["id"])
        return result
