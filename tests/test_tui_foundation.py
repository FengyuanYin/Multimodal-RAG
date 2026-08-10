import pytest

from agentic_rag.tui.config import SecretStore
from agentic_rag.tui.paths import AutoMemoryPaths
from agentic_rag.tui.security import ensure_within, redact, safe_filename
from agentic_rag.tui.storage import StateRepository


def test_paths_are_isolated_and_created(tmp_path):
    paths = AutoMemoryPaths.resolve(tmp_path / "automemory")
    assert paths.state_db.parent.is_dir()
    assert paths.knowledge_db.parent.is_dir()
    assert paths.contains(paths.exports_dir / "result.json")
    assert not paths.contains(tmp_path.parent / "outside")


def test_state_roundtrip_and_stream_recovery(tmp_path):
    path = tmp_path / "state.db"
    state = StateRepository(path)
    conversation = state.create_conversation("Test")
    state.append_message(conversation.id, "assistant", "partial", "direct", "streaming")
    state.create_memory("Remember this")
    state.close()
    reopened = StateRepository(path)
    assert reopened.list_messages(conversation.id)[0].status == "interrupted"
    assert reopened.list_memories(enabled_only=True)[0].content == "Remember this"
    reopened.close()


def test_secrets_are_runtime_only_and_redacted(tmp_path):
    secret = "sk-test-super-secret"
    store = SecretStore(environ={})
    store.set("llm_api_key", secret)
    state = StateRepository(tmp_path / "state.db")
    state.save_settings({"llm_model": "model"})
    with pytest.raises(ValueError):
        state.save_settings({"api_key": secret})
    state.close()
    assert secret not in (tmp_path / "state.db").read_bytes().decode("latin1")
    assert secret not in redact(f"Bearer {secret} api_key={secret}", store.values_for_redaction())


def test_safe_exports_reject_traversal(tmp_path):
    root = tmp_path / "exports"
    root.mkdir()
    assert safe_filename("../bad:name.txt") == "bad_name.txt"
    with pytest.raises(ValueError):
        ensure_within(root, tmp_path / "outside.txt")
