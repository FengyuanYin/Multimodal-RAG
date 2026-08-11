from __future__ import annotations

from copy import deepcopy
from io import StringIO
from types import SimpleNamespace

import pytest

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.config import AutoMemoryConfig
from agentic_rag.cli.errors import ConfigurationError
from agentic_rag.cli.setup_wizard import SetupApplier, SetupWizard


class FakeCredentials:
    def __init__(self):
        self.values = {}

    def configured(self, name):
        return bool(self.values.get(name))

    def source(self, name):
        return "stored" if self.configured(name) else "not-configured"

    def get(self, name):
        return self.values.get(name, "")

    def get_persisted(self, name):
        return self.values.get(name)

    def set(self, name, value, persist=True):
        self.values[name] = value

    def delete(self, name):
        return self.values.pop(name, None) is not None

    def restore_persisted(self, name, value):
        if value is None:
            self.values.pop(name, None)
        else:
            self.values[name] = value


class FakeConfigStore:
    def __init__(self, config, fail=False):
        self.saved = deepcopy(config)
        self.fail = fail

    def save(self, config):
        if self.fail:
            self.fail = False
            raise OSError("injected")
        self.saved = AutoMemoryConfig.from_dict(config.to_dict())


class ScriptedTerminal:
    interactive = True

    def __init__(self, choices, forms, secrets, confirms):
        self.choices, self.forms, self.secrets, self.confirms = list(choices), list(forms), list(secrets), list(confirms)
        self.events = []

    def choose(self, *args, **kwargs):
        return self.choices.pop(0)

    def read_form_value(self, prompt, default=""):
        value = self.forms.pop(0)
        return value or default

    def read_secret(self, prompt):
        return self.secrets.pop(0)

    def confirm(self, *args, **kwargs):
        return self.confirms.pop(0)

    def emit(self, event):
        self.events.append(event)


def make_context(config=None):
    config = config or AutoMemoryConfig()
    credentials = FakeCredentials()
    store = FakeConfigStore(config)
    ctx = SimpleNamespace(config=config, credentials=credentials, config_store=store, paths=SimpleNamespace(backups_dir="backups"), connectivity=SimpleNamespace(test_services=lambda *args: []))
    ctx.reload_services = lambda candidate: setattr(ctx, "config", AutoMemoryConfig.from_dict(candidate.to_dict()))
    return ctx


def test_only_llm_setup_commits_after_confirmation() -> None:
    ctx = make_context()
    terminal = ScriptedTerminal(
        ["configure", "deepseek", "skip", "skip", "skip", "skip", "skip"],
        ["", ""], ["new-secret"], [False, True],
    )
    result = SetupWizard(ctx, terminal, CancellationToken()).run()
    assert result.ok
    assert ctx.config.llm.base_url == "https://api.deepseek.com"
    assert ctx.config.llm.model == "deepseek-v4-flash"
    assert ctx.credentials.values["llm_api_key"] == "new-secret"


def test_cancel_has_no_side_effects() -> None:
    ctx = make_context()
    before = ctx.config.to_dict()
    result = SetupWizard(ctx, ScriptedTerminal(["cancel"], [], [], []), CancellationToken()).run()
    assert not result.ok
    assert ctx.config.to_dict() == before
    assert not ctx.credentials.values


def test_commit_failure_restores_config_and_credentials() -> None:
    ctx = make_context()
    ctx.credentials.values["llm_api_key"] = "old-secret"
    ctx.config_store.fail = True
    draft = SimpleNamespace(config=AutoMemoryConfig(), secrets={"llm_api_key": "new-secret"})
    draft.config.llm.model = "changed-model"
    with pytest.raises(ConfigurationError):
        SetupApplier.commit(ctx, draft)
    assert ctx.credentials.values["llm_api_key"] == "old-secret"
    assert ctx.config.llm.model != "changed-model"


def test_siliconflow_reranker_can_reuse_staged_llm_key() -> None:
    ctx = make_context()
    terminal = ScriptedTerminal(
        ["configure", "siliconflow", "skip", "skip", "configure", "siliconflow", "skip", "skip"],
        ["", "", "", ""],
        ["shared-siliconflow-key"],
        [True, False, True],
    )

    result = SetupWizard(ctx, terminal, CancellationToken()).run()

    assert result.ok
    assert ctx.config.reranker.base_url == "https://api.siliconflow.cn/v1"
    assert ctx.config.reranker.model == "BAAI/bge-reranker-v2-m3"
    assert ctx.credentials.values["reranker_api_key"] == "shared-siliconflow-key"
    assert all("shared-siliconflow-key" not in event.text for event in terminal.events)
