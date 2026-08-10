"""Image metadata parsing without local OCR or models."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path

from ..cancellation import CancellationToken
from ..errors import ConfigurationError
from ..models import ParsedDocument


def parse_image(path: Path, cancel: CancellationToken) -> ParsedDocument:
    cancel.checkpoint()
    content = path.read_bytes()
    if not content or len(content) > 10 * 1024 * 1024:
        raise ConfigurationError("Image is empty or exceeds the 10MB limit")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    checksum = hashlib.sha256(content).hexdigest()
    media = [{
        "id": f"image_{checksum[:16]}", "page": 1, "type": "image", "label": "figure1",
        "caption": "", "data": base64.b64encode(content).decode("ascii"), "mime_type": mime,
        "checksum": checksum, "quality": "exact",
    }]
    cancel.checkpoint()
    return ParsedDocument(path.name, [{"page": 1, "text": f"[Image: {path.name}]"}], media, "local_image_metadata")
