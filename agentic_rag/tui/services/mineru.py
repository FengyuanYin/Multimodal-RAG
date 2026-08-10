"""Official and self-hosted MinerU clients for AutoMemory."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import mimetypes
from pathlib import Path, PurePosixPath
import time
import zipfile

import httpx

from ..events import CancelToken, EventCallback, JobProgress
from ..models import ParsedExternalDocument
from ..security import safe_filename, validate_public_url


class MinerUService:
    API_BASE = "https://mineru.net/api/v4"
    MAX_FILE = 50 * 1024 * 1024
    MAX_ARCHIVE = 64 * 1024 * 1024
    TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=120.0, pool=15.0)

    @staticmethod
    def _progress(emit: EventCallback | None, job_id: str, phase: str, message: str) -> None:
        if emit:
            emit(JobProgress(job_id, "mineru", phase, message))

    @staticmethod
    async def _sleep(seconds: float, cancel: CancelToken) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            cancel.checkpoint()
            await asyncio.sleep(min(0.25, deadline - time.monotonic()))

    def _validate_pdf(self, path: Path) -> bytes:
        path = path.resolve()
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise ValueError("MinerU requires a readable PDF file")
        size = path.stat().st_size
        if not size or size > self.MAX_FILE:
            raise ValueError("PDF is empty or exceeds the 50MB limit")
        return path.read_bytes()

    async def parse_official(self, path: Path, api_key: str, emit: EventCallback | None = None, cancel: CancelToken | None = None, job_id: str = "mineru") -> ParsedExternalDocument:
        cancel = cancel or CancelToken()
        if not api_key.strip():
            raise ValueError("MinerU API key is required")
        body = self._validate_pdf(path)
        cancel.checkpoint()
        auth = {"Authorization": f"Bearer {api_key.strip()}"}
        async with httpx.AsyncClient(timeout=self.TIMEOUT, follow_redirects=True) as client:
            self._progress(emit, job_id, "create", "Creating MinerU batch")
            create = await client.post(f"{self.API_BASE}/file-urls/batch", headers=auth, json={
                "files": [{"name": safe_filename(path.name, "document.pdf"), "data_id": path.stem[:80]}],
                "model_version": "vlm", "enable_formula": True, "enable_table": True, "language": "ch",
            })
            payload = self._json(create, "MinerU create-task response")
            if create.status_code >= 400 or payload.get("code") not in (None, 0):
                raise RuntimeError(str(payload.get("msg") or payload.get("message") or f"MinerU create failed (HTTP {create.status_code})")[:300])
            data = payload.get("data") or {}
            batch_id, upload_urls = data.get("batch_id"), data.get("file_urls") or []
            if not batch_id or not upload_urls:
                raise RuntimeError("MinerU did not return a batch ID and upload URL")
            upload_url = validate_public_url(str(upload_urls[0]))
            cancel.checkpoint()
            self._progress(emit, job_id, "upload", "Uploading PDF")
            upload = await client.put(upload_url, content=body)
            if upload.status_code >= 400:
                raise RuntimeError(f"MinerU PDF upload failed (HTTP {upload.status_code})")
            deadline, failures, item = time.monotonic() + 300, 0, None
            while time.monotonic() < deadline:
                await self._sleep(2, cancel)
                self._progress(emit, job_id, "poll", "Waiting for MinerU extraction")
                try:
                    status = await client.get(f"{self.API_BASE}/extract-results/batch/{batch_id}", headers=auth)
                    status_payload = self._json(status, "MinerU status response")
                    failures = 0
                except httpx.HTTPError:
                    failures += 1
                    if failures >= 3:
                        raise RuntimeError("MinerU status polling failed after retries")
                    await self._sleep(min(2 ** failures, 5), cancel)
                    continue
                if status.status_code >= 400 or status_payload.get("code") not in (None, 0):
                    raise RuntimeError(str(status_payload.get("msg") or f"MinerU status failed (HTTP {status.status_code})")[:300])
                results = (status_payload.get("data") or {}).get("extract_result") or []
                item = results[0] if results else None
                state = str((item or {}).get("state", "")).lower()
                if state == "done":
                    break
                if state == "failed":
                    raise RuntimeError(str((item or {}).get("err_msg") or "MinerU extraction failed")[:300])
            else:
                raise TimeoutError("MinerU extraction exceeded five minutes")
            archive_url = (item or {}).get("full_zip_url")
            if not archive_url:
                raise RuntimeError("MinerU completed without a result archive URL")
            archive_url = validate_public_url(str(archive_url))
            self._progress(emit, job_id, "download", "Downloading MinerU result")
            archive = await self._download(client, archive_url, cancel)
        self._progress(emit, job_id, "normalize", "Normalizing pages and media")
        return self.normalize_archive(archive, path.name, "mineru_official_v4")

    async def parse_selfhosted(self, path: Path, endpoint: str, api_key: str = "", emit: EventCallback | None = None, cancel: CancelToken | None = None, job_id: str = "mineru") -> ParsedExternalDocument:
        cancel = cancel or CancelToken()
        body = self._validate_pdf(path)
        endpoint = validate_public_url(endpoint, allow_private=True).rstrip("/")
        headers = {"X-File-Name": safe_filename(path.name, "document.pdf"), "Content-Type": "application/pdf"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._progress(emit, job_id, "upload", "Sending PDF to self-hosted MinerU")
        async with httpx.AsyncClient(timeout=self.TIMEOUT, follow_redirects=True) as client:
            response = await client.post(endpoint, headers=headers, content=body)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "zip" in content_type:
                if len(response.content) > self.MAX_ARCHIVE:
                    raise ValueError("MinerU result exceeds the 64MB limit")
                return self.normalize_archive(response.content, path.name, "mineru_selfhosted")
            payload = self._json(response, "self-hosted MinerU response")
        return self.normalize_payload(payload, path.name, "mineru_selfhosted")

    async def _download(self, client: httpx.AsyncClient, url: str, cancel: CancelToken) -> bytes:
        last_error = None
        for attempt in range(3):
            cancel.checkpoint()
            try:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    chunks, total = [], 0
                    async for chunk in response.aiter_bytes():
                        cancel.checkpoint()
                        total += len(chunk)
                        if total > self.MAX_ARCHIVE:
                            raise ValueError("MinerU result exceeds the 64MB limit")
                        chunks.append(chunk)
                    return b"".join(chunks)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < 2:
                    await self._sleep(2 ** attempt, cancel)
        raise RuntimeError(f"MinerU result download failed ({type(last_error).__name__})")

    @staticmethod
    def _json(response: httpx.Response, label: str) -> dict:
        try:
            value = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"{label} has an unsupported shape")
        return value

    def normalize_archive(self, archive: bytes, title: str, parser: str) -> ParsedExternalDocument:
        try:
            bundle = zipfile.ZipFile(io.BytesIO(archive))
        except zipfile.BadZipFile as exc:
            raise ValueError("MinerU result is not a valid ZIP archive") from exc
        infos = [info for info in bundle.infolist() if not info.is_dir()]
        if sum(info.file_size for info in infos) > self.MAX_ARCHIVE:
            raise ValueError("MinerU expanded result exceeds the 64MB limit")
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("MinerU archive contains an unsafe path")
        names = {info.filename: info for info in infos}
        content_name = next((name for name in names if name.endswith("_content_list.json")), None)
        pages: dict[int, list[str]] = {}
        media = []
        if content_name:
            items = json.loads(bundle.read(content_name).decode("utf-8"))
            for sequence, item in enumerate(items if isinstance(items, list) else [], 1):
                page = max(1, int(item.get("page_idx", 0)) + 1)
                kind = str(item.get("type") or "text").lower()
                text = str(item.get("text") or item.get("table_body") or item.get("latex") or "").strip()
                caption_value = item.get("img_caption") or item.get("table_caption") or ""
                caption = " ".join(map(str, caption_value)) if isinstance(caption_value, list) else str(caption_value)
                if caption and caption not in text:
                    text = f"{caption}\n{text}".strip()
                if text:
                    pages.setdefault(page, []).append(text)
                image_path = str(item.get("img_path") or "").replace("\\", "/")
                if kind == "image" and image_path:
                    candidate = str(PurePosixPath(content_name).parent / image_path)
                    info = names.get(candidate) or names.get(image_path)
                    if info and info.file_size <= 4 * 1024 * 1024:
                        raw = bundle.read(info)
                        media.append({"id": f"mineru_image_{sequence}_p{page}", "page": page, "label": f"图{sequence}", "type": "image", "caption": caption, "data": base64.b64encode(raw).decode("ascii"), "mime_type": mimetypes.guess_type(info.filename)[0] or "image/png", "quality": "exact", "extraction_method": parser})
        if not pages:
            markdown = [name for name in names if name.lower().endswith(".md")]
            if not markdown:
                raise ValueError("MinerU result contains no Markdown or content list")
            text = bundle.read(max(markdown, key=lambda name: names[name].file_size)).decode("utf-8", errors="replace").strip()
            if not text:
                raise ValueError("MinerU Markdown result is empty")
            pages[1] = [text]
        return ParsedExternalDocument(title, [{"page": page, "text": "\n\n".join(parts)} for page, parts in sorted(pages.items())], media, parser)

    @staticmethod
    def normalize_payload(payload: dict, title: str, parser: str) -> ParsedExternalDocument:
        if isinstance(payload.get("pages"), list):
            pages = [{"page": int(item.get("page", 1)), "text": str(item.get("text") or item.get("content") or "")} for item in payload["pages"]]
        elif str(payload.get("text") or "").strip():
            pages = [{"page": 1, "text": str(payload["text"])}]
        else:
            raise ValueError("self-hosted MinerU returned an unsupported result")
        pages = [item for item in pages if item["text"].strip()]
        if not pages:
            raise ValueError("self-hosted MinerU returned empty text")
        return ParsedExternalDocument(title, pages, list(payload.get("media") or []), parser)
