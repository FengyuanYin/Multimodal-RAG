from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.errors import UpstreamError
from agentic_rag.cli.services.connectivity import ConnectionTester
from agentic_rag.cli.terminal import PlainTerminal


class Credentials:
    def get(self, name):
        return "test-secret"

    def redaction_values(self):
        return ("test-secret",)


class Client:
    def probe_chat(self, cancel):
        cancel.checkpoint()

    def probe_embedding(self, cancel):
        cancel.checkpoint()

    def probe(self, cancel, **kwargs):
        cancel.checkpoint()
        return True


def context() -> SimpleNamespace:
    return SimpleNamespace(
        llm_client=Client(), embedding_client=Client(), vlm_client=Client(), reranker_client=Client(),
        web_client=SimpleNamespace(search=lambda *args, **kwargs: []), credentials=Credentials(),
        config=SimpleNamespace(mineru_mode="official", web_provider="duckduckgo"),
        mineru_client=lambda: SimpleNamespace(probe=lambda *args, **kwargs: False, close=lambda: None),
    )


def test_mixed_connectivity_results_are_independent() -> None:
    ctx = context()
    error = UpstreamError("bad credential test-secret")
    error.code = "UPSTREAM_AUTH"
    ctx.embedding_client.probe_embedding = lambda cancel: (_ for _ in ()).throw(error)
    stdout = StringIO()
    results = ConnectionTester(ctx).test_services({"llm", "embedding", "mineru"}, PlainTerminal(stdout=stdout, stderr=StringIO()), CancellationToken())
    assert [item.status for item in results] == ["success", "auth_error", "reachable_unverified"]
    assert "test-secret" not in stdout.getvalue()


@pytest.mark.parametrize(
    ("code", "status"),
    [("UPSTREAM_AUTH", "auth_error"), ("UPSTREAM_RATE_LIMIT", "rate_limited"), ("UPSTREAM_NETWORK", "network_error"), ("UPSTREAM_MODEL", "model_error")],
)
def test_error_classification(code: str, status: str) -> None:
    error = UpstreamError("failure")
    error.code = code
    assert ConnectionTester._classify(error)[0] == status
