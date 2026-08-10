"""Security boundaries for local network capture, logs, and exports."""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_SECRET_KEY = re.compile(r"(api[_-]?key|authorization|token|secret|password)", re.I)
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+\-/=]{8,}")
_KEY_VALUE = re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)[^\s,;]+")
_SIGNED_QUERY_KEYS = {"signature", "sig", "token", "x-amz-signature", "x-oss-signature", "authorization"}


def looks_secret_key(name: str) -> bool:
    return bool(_SECRET_KEY.search(name))


def redact(value: object, secrets: tuple[str, ...] = ()) -> str:
    text = str(value)
    text = _BEARER.sub("Bearer [REDACTED]", text)
    text = _KEY_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    for secret in secrets:
        if secret and len(secret) >= 6:
            text = text.replace(secret, "[REDACTED]")
    try:
        parsed = urlsplit(text)
        if parsed.scheme in {"http", "https"} and parsed.query:
            query = [(k, "[REDACTED]" if k.lower() in _SIGNED_QUERY_KEYS else v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)]
            text = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))
    except ValueError:
        pass
    return text


def validate_public_url(url: str, *, allow_private: bool = False) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    if allow_private:
        return url.strip()
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("target hostname could not be resolved") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not address.is_global:
            raise ValueError("private or reserved network targets are blocked")
    return url.strip()


def safe_filename(name: str, fallback: str = "export") -> str:
    value = Path(name.replace("\\", "/")).name
    value = re.sub(r"[^\w.()\- ]+", "_", value, flags=re.UNICODE).strip(" .")
    return value[:180] or fallback


def ensure_within(root: Path, destination: Path) -> Path:
    root = root.resolve()
    destination = destination.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("destination must stay inside the configured export directory") from exc
    return destination
