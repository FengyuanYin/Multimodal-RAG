"""Stable prompt construction for full-document workspaces."""

from __future__ import annotations

import hashlib
import json


PROMPT_VERSION = "document-workspace-v1"
SYSTEM_PROMPT = """You are AutoMemory's full-document analyst. The complete immutable Markdown below is the primary document fact source. Answer with the stronger main LLM. Do not claim to have inspected an image unless a read_image tool result is present. History summaries and answer previews are explicitly labeled and are not original/full content. Use tools only for this document workspace. Never invent file contents."""


class DocumentContextBuilder:
    def build_fixed_prefix(self, document: dict, artifact: dict, markdown: str, media: list[dict]) -> list[dict]:
        media_manifest = []
        for item in sorted(media, key=lambda value: value["id"]):
            metadata = item.get("metadata") or {}
            media_manifest.append({
                "media_id": item["id"],
                "markdown_reference": metadata.get("markdown_reference", ""),
                "label": item.get("label", ""),
                "page": item.get("page"),
                "caption": item.get("caption", ""),
            })
        identity = {"prompt_version": PROMPT_VERSION, "document_id": document["id"], "title": document["title"], "markdown_sha256": artifact["checksum"], "media": media_manifest}
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "DOCUMENT_IDENTITY\n" + json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
            {"role": "system", "content": "<AUTOMEMORY_FULL_MARKDOWN immutable=\"true\">\n" + markdown + "\n</AUTOMEMORY_FULL_MARKDOWN>"},
        ]

    def build_variable_suffix(self, events: list[dict], question: str) -> list[dict]:
        suffix = []
        for item in events:
            if not item.get("content"):
                continue
            if item["role"] == "tool":
                suffix.append({"role":"system", "content":"<AUTOMEMORY_PAST_TOOL_RESULT>\n" + item["content"] + "\n</AUTOMEMORY_PAST_TOOL_RESULT>"})
            elif item["role"] in {"user", "assistant", "system"}:
                suffix.append({"role":item["role"], "content":item["content"]})
        suffix.append({"role": "user", "content": question})
        return suffix

    @staticmethod
    def prefix_fingerprint(messages: list[dict]) -> str:
        raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
