"""Credential redaction, URL validation, and filesystem guards."""

from __future__ import annotations

import ipaddress
from pathlib import Path
import re
import socket
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .errors import SecurityError


_SECRET_NAME = re.compile(r"(api[_-]?key|authorization|bearer|token|secret|password)", re.I)
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+\-/=]{8,}")
_KEY_VALUE = re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)[^\s,;]+")
_KEY_SHAPE = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|[A-Za-z0-9_-]{32,})")
_SIGNED_QUERY_KEYS = {"signature", "sig", "token", "api_key", "x-amz-signature", "x-oss-signature", "authorization"}


def looks_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME.search(name))


def looks_sensitive_text(value: str) -> bool:
    text = value.strip()
    return bool(_BEARER.search(text) or _KEY_VALUE.search(text) or _KEY_SHAPE.search(text))


def redact(value: object, secrets: tuple[str, ...] = ()) -> str:
    text = str(value)
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _KEY_VALUE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text)
    for secret in secrets:
        if secret and len(secret) >= 6:
            text = text.replace(secret, "[REDACTED]")
    try:
        parsed = urlsplit(text)
        if parsed.scheme in {"http", "https"} and parsed.query:
            query = [(key, "[REDACTED]" if key.lower() in _SIGNED_QUERY_KEYS else item) for key, item in parse_qsl(parsed.query, keep_blank_values=True)]
            text = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))
    except ValueError:
        pass
    return text


def validate_http_url(url: str, *, allow_private: bool = False, resolve: bool = True) -> str:
    raw = url.strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SecurityError("URL must use http or https")
    if parsed.username or parsed.password:
        raise SecurityError("URLs containing credentials are not allowed")
    if allow_private or not resolve:
        return raw
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SecurityError("Target hostname could not be resolved") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not address.is_global:
            raise SecurityError("Private, loopback, or reserved network targets are blocked")
    return raw


def safe_filename(name: str, fallback: str = "export") -> str:
    value = Path(name.replace("\\", "/")).name
    value = re.sub(r"[^\w.()\- ]+", "_", value, flags=re.UNICODE).strip(" .")
    return value[:180] or fallback


def ensure_within(root: Path, destination: Path) -> Path:
    root, destination = root.resolve(), destination.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise SecurityError("Destination must stay inside the configured export directory") from exc
    return destination
