"""Official and self-hosted cloud MinerU document parsing."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
from pathlib import Path, PurePosixPath
import time
import zipfile

from ..cancellation import CancellationToken
from ..errors import ConfigurationError, UpstreamError
from ..models import EventKind, OutputEvent, ParsedDocument
from ..security import safe_filename, validate_http_url
from .transport import HttpTransport


class MinerUClient:
    MAX_PDF = 50 * 1024 * 1024
    MAX_ARCHIVE = 64 * 1024 * 1024

    def __init__(self, base_url: str, api_key: str, *, transport: HttpTransport | None = None, allow_private: bool = False) -> None:
        self.base_url, self.api_key, self.allow_private = base_url.rstrip("/"), api_key, allow_private
        self.transport = transport or HttpTransport("MinerU", 120.0, 2, secrets=(api_key,))

    @staticmethod
    def _emit(output, task_id: str, phase: str, text: str) -> None:
        if output:
            output.emit(OutputEvent(EventKind.PROGRESS, text=text, task_id=task_id, phase=phase))

    @classmethod
    def _read_pdf(cls, path: Path) -> bytes:
        path = path.resolve()
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise ConfigurationError("MinerU requires an existing PDF file")
        size = path.stat().st_size
        if size <= 0 or size > cls.MAX_PDF:
            raise ConfigurationError("PDF is empty or exceeds the 50MB MinerU limit")
        return path.read_bytes()

    def parse_official(self, path: Path, output, cancel: CancellationToken, task_id: str) -> ParsedDocument:
        if not self.api_key:
            raise ConfigurationError("MinerU API key is not configured")
        body, headers = self._read_pdf(path), {"Authorization": f"Bearer {self.api_key}"}
        self._emit(output, task_id, "create", "Creating MinerU batch")
        created = self.transport.request_json(
            "POST", f"{self.base_url}/file-urls/batch", headers=headers,
            json_body={"files": [{"name": safe_filename(path.name, "document.pdf"), "data_id": path.stem[:80]}], "model_version": "vlm", "enable_formula": True, "enable_table": True, "language": "ch"},
            cancel=cancel,
        )
        if created.get("code") not in (None, 0):
            raise UpstreamError(str(created.get("msg") or "MinerU create task failed")[:300])
        data = created.get("data") or {}
        batch_id, upload_urls = data.get("batch_id"), data.get("file_urls") or []
        if not batch_id or not upload_urls:
            raise UpstreamError("MinerU did not return a batch ID and upload URL")
        upload_url = validate_http_url(str(upload_urls[0]))
        self._emit(output, task_id, "upload", "Uploading PDF")
        with self.transport.stream("PUT", upload_url, content=body, cancel=cancel) as response:
            response.read()
        deadline, result_item = time.monotonic() + 300, None
        while time.monotonic() < deadline:
            self.transport._sleep(2.0, cancel)
            self._emit(output, task_id, "poll", "Waiting for MinerU extraction")
            status = self.transport.request_json("GET", f"{self.base_url}/extract-results/batch/{batch_id}", headers=headers, cancel=cancel, idempotent=True)
            results = (status.get("data") or {}).get("extract_result") or []
            result_item = results[0] if results else None
            state = str((result_item or {}).get("state", "")).lower()
            if state == "done":
                break
            if state == "failed":
                raise UpstreamError(str((result_item or {}).get("err_msg") or "MinerU extraction failed")[:300])
        else:
            error = UpstreamError("MinerU extraction exceeded five minutes")
            error.code = "MINERU_TIMEOUT"
            raise error
        archive_url = validate_http_url(str((result_item or {}).get("full_zip_url") or ""))
        self._emit(output, task_id, "download", "Downloading MinerU result")
        archive = self._download(archive_url, cancel)
        self._emit(output, task_id, "normalize", "Normalizing MinerU pages and media")
        return self.normalize_archive(archive, path.name, "mineru_official")

    def parse_selfhosted(self, path: Path, output, cancel: CancellationToken, task_id: str) -> ParsedDocument:
        body = self._read_pdf(path)
        endpoint = validate_http_url(self.base_url, allow_private=self.allow_private)
        headers = {"Content-Type": "application/pdf", "X-File-Name": safe_filename(path.name, "document.pdf")}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._emit(output, task_id, "upload", "Sending PDF to self-hosted MinerU")
        with self.transport.stream("POST", endpoint, headers=headers, content=body, cancel=cancel) as response:
            content_type = response.headers.get("content-type", "").lower()
            raw = response.read()
        if len(raw) > self.MAX_ARCHIVE:
            raise UpstreamError("MinerU response exceeded the 64MB limit")
        if "zip" in content_type:
            return self.normalize_archive(raw, path.name, "mineru_selfhosted")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UpstreamError("Self-hosted MinerU returned neither ZIP nor valid JSON") from exc
        return self.normalize_payload(payload, path.name, "mineru_selfhosted")

    def _download(self, url: str, cancel: CancellationToken) -> bytes:
        with self.transport.stream("GET", url, cancel=cancel) as response:
            chunks, total = [], 0
            for chunk in response.iter_bytes():
                cancel.checkpoint()
                total += len(chunk)
                if total > self.MAX_ARCHIVE:
                    raise UpstreamError("MinerU archive exceeded the 64MB limit")
                chunks.append(chunk)
        return b"".join(chunks)

    def normalize_archive(self, archive: bytes, title: str, parser: str) -> ParsedDocument:
        try:
            bundle = zipfile.ZipFile(io.BytesIO(archive))
        except zipfile.BadZipFile as exc:
            raise UpstreamError("MinerU result is not a valid ZIP archive") from exc
        infos = [item for item in bundle.infolist() if not item.is_dir()]
        if sum(item.file_size for item in infos) > self.MAX_ARCHIVE:
            raise UpstreamError("MinerU expanded archive exceeded the 64MB limit")
        for item in infos:
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise UpstreamError("MinerU archive contains an unsafe path")
        names = {item.filename: item for item in infos}
        content_name = next((name for name in names if name.endswith("_content_list.json")), None)
        pages: dict[int, list[str]] = {}
        media = []
        if content_name:
            try:
                items = json.loads(bundle.read(content_name).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise UpstreamError("MinerU content list is invalid") from exc
            for sequence, item in enumerate(items if isinstance(items, list) else [], 1):
                page = max(1, int(item.get("page_idx", 0)) + 1)
                text = str(item.get("text") or item.get("table_body") or item.get("latex") or "").strip()
                caption_value = item.get("img_caption") or item.get("table_caption") or ""
                caption = " ".join(map(str, caption_value)) if isinstance(caption_value, list) else str(caption_value)
                if caption and caption not in text:
                    text = f"{caption}\n{text}".strip()
                if text:
                    pages.setdefault(page, []).append(text)
                image_path = str(item.get("img_path") or "").replace("\\", "/")
                if image_path:
                    candidate = str(PurePosixPath(content_name).parent / image_path)
                    info = names.get(candidate) or names.get(image_path)
                    if info and info.file_size <= 8 * 1024 * 1024:
                        raw = bundle.read(info)
                        media.append({"id": f"figure_{sequence}_p{page}", "page": page, "type": "image", "label": f"figure{sequence}", "caption": caption, "data": base64.b64encode(raw).decode("ascii"), "mime_type": mimetypes.guess_type(info.filename)[0] or "image/png", "quality": "exact"})
        if not pages:
            markdown = [name for name in names if name.lower().endswith(".md")]
            if not markdown:
                raise UpstreamError("MinerU archive contains no readable content")
            text = bundle.read(max(markdown, key=lambda name: names[name].file_size)).decode("utf-8", errors="replace").strip()
            if not text:
                raise UpstreamError("MinerU Markdown result is empty")
            pages[1] = [text]
        return ParsedDocument(title, [{"page": page, "text": "\n\n".join(parts)} for page, parts in sorted(pages.items())], media, parser)

    @staticmethod
    def normalize_payload(payload: dict, title: str, parser: str) -> ParsedDocument:
        if not isinstance(payload, dict):
            raise UpstreamError("Self-hosted MinerU returned an unsupported response")
        if isinstance(payload.get("pages"), list):
            pages = [{"page": int(item.get("page", 1)), "text": str(item.get("text") or item.get("content") or "")} for item in payload["pages"]]
        elif str(payload.get("text") or "").strip():
            pages = [{"page": 1, "text": str(payload["text"])}]
        else:
            raise UpstreamError("Self-hosted MinerU returned no readable content")
        pages = [item for item in pages if item["text"].strip()]
        if not pages:
            raise UpstreamError("Self-hosted MinerU returned empty content")
        return ParsedDocument(title, pages, list(payload.get("media") or []), parser)

    def close(self) -> None:
        self.transport.close()
