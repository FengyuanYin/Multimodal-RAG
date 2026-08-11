from __future__ import annotations

from io import StringIO

import pytest

from agentic_rag.cli.credentials import CredentialStore
from agentic_rag.cli.errors import SecurityError, UsageError
from agentic_rag.cli.security import redact, validate_http_url
from agentic_rag.cli.terminal import PlainTerminal
from agentic_rag.cli.terminal import InteractiveTerminal


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


def test_prompt_toolkit_form_and_secret_use_private_session() -> None:
    calls = []

    class Session:
        def prompt(self, prompt, **kwargs):
            calls.append(kwargs)
            return "value"

    terminal = InteractiveTerminal.__new__(InteractiveTerminal)
    PlainTerminal.__init__(terminal, stdin=StringIO(), stdout=StringIO(), stderr=StringIO(), interactive=True)
    terminal._session = object()
    terminal._private_session = Session()
    assert terminal.read_form_value("Model") == "value"
    assert terminal.read_secret("Key: ") == "value"
    assert "add_history" not in calls[0]
    assert "add_history" not in calls[1] and calls[1]["is_password"] is True


def test_prompt_toolkit_private_session_discards_history(tmp_path, monkeypatch) -> None:
    sessions = []

    class Session:
        def __init__(self, **kwargs):
            self.history = kwargs["history"]
            sessions.append(self)

    monkeypatch.setattr("prompt_toolkit.PromptSession", Session)
    terminal = InteractiveTerminal(tmp_path / "history", lambda _: [], no_color=True)
    assert len(sessions) == 2
    assert terminal._private_session.history.__class__.__name__ == "DummyHistory"
