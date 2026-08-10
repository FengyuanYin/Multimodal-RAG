from __future__ import annotations

from io import StringIO

import pytest

from agentic_rag.cli.credentials import CredentialStore
from agentic_rag.cli.errors import SecurityError, UsageError
from agentic_rag.cli.security import redact, validate_http_url
from agentic_rag.cli.terminal import PlainTerminal


def test_environment_credential_override_is_not_echoed() -> None:
    secret = "sk-super-secret-value-123456789"
    store = CredentialStore({"AUTOMEMORY_LLM_API_KEY": secret})
    assert store.get("llm_api_key") == secret
    assert store.source("llm_api_key") == "environment:AUTOMEMORY_LLM_API_KEY"
    assert secret not in redact(f"request failed with {secret}", store.redaction_values())


def test_private_urls_and_url_credentials_are_blocked(monkeypatch) -> None:
    monkeypatch.setattr("socket.getaddrinfo", lambda *args, **kwargs: [(2, 1, 6, "", ("127.0.0.1", 80))])
    with pytest.raises(SecurityError):
        validate_http_url("http://localhost/internal")
    with pytest.raises(SecurityError):
        validate_http_url("https://user:pass@example.com/")


def test_noninteractive_secret_prompt_fails_safely() -> None:
    terminal = PlainTerminal(stdin=StringIO(), stdout=StringIO(), stderr=StringIO())
    with pytest.raises(UsageError):
        terminal.read_secret("secret: ")
