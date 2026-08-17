from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.config import AutoMemoryConfig, validate_config
from agentic_rag.cli.models import ChunkRecord, DocumentRecord, MediaRecord, ModelStreamEvent, ParsedDocument
from agentic_rag.cli.paths import AutoMemoryPaths
from agentic_rag.cli.services.context_budget import ConservativeTokenEstimator, ContextBudgetService
from agentic_rag.cli.services.document_artifacts import DocumentArtifactService
from agentic_rag.cli.services.document_context import DocumentContextBuilder, PROMPT_VERSION
from agentic_rag.cli.services.document_workspace_chat import DocumentWorkspaceChatService
from agentic_rag.cli.services.long_response import LongResponseService
from agentic_rag.cli.services.workspace_files import WorkspaceFileService
from agentic_rag.cli.storage import KnowledgeRepository, StateRepository
from agentic_rag.cli.terminal import PlainTerminal


class FakeLLM:
    profile_fingerprint = "main-model"

    def __init__(self):
        self.requests = []

    def stream_chat_events(self, messages, tools, cancel, **kwargs):
        self.requests.append(messages)
        yield ModelStreamEvent("text_delta", text="document answer")
        yield ModelStreamEvent("usage", usage={"prompt_tokens": 100, "completion_tokens": 2, "cached_tokens": 50})


class NoCompaction:
    def compact(self, *args, **kwargs):
        raise AssertionError("compaction was not expected")


class NoImages:
    def analyze(self, *args, **kwargs):
        raise AssertionError("VLM must not run without a tool call")


def _fixture(tmp_path: Path):
    paths = AutoMemoryPaths.resolve(tmp_path / "home")
    knowledge = KnowledgeRepository(paths.knowledge_db, paths.backups_dir)
    state = StateRepository(paths.state_db, paths.backups_dir)
    knowledge.commit_document(DocumentRecord("doc","fp","Paper","paper.pdf","pdf","default","mineru_official",1,"ready"), [ChunkRecord("chunk","doc",1,0,"evidence")], [])
    artifacts = DocumentArtifactService(knowledge, paths.knowledge_assets_dir)
    artifact = artifacts.save_markdown("doc", ParsedDocument("Paper", [{"page":1,"text":"evidence"}], [], "mineru_official", "HEAD\nMIDDLE\nTAIL", "mineru_original"))
    workspace = state.workspaces.open_or_create(state.ensure_active_conversation(), "doc", artifact.id, artifact.checksum, "main-model", PROMPT_VERSION)
    config = AutoMemoryConfig()
    estimator = ConservativeTokenEstimator()
    files = WorkspaceFileService(state.workspaces, paths.workspaces_dir, paths.exports_dir, estimator)
    return paths, knowledge, state, artifacts, workspace, config, files, estimator


def test_full_document_chat_keeps_complete_stable_prefix_without_vlm(tmp_path: Path) -> None:
    paths, knowledge, state, artifacts, workspace, config, files, estimator = _fixture(tmp_path)
    llm = FakeLLM(); builder = DocumentContextBuilder()
    service = DocumentWorkspaceChatService(state, knowledge, artifacts, llm, builder, ContextBudgetService(config, estimator), NoCompaction(), files, NoImages(), LongResponseService(config, files, estimator), config)
    output = PlainTerminal(stdout=StringIO(), stderr=StringIO())
    service.stream(workspace["id"], "first", output, CancellationToken())
    service.stream(workspace["id"], "second", output, CancellationToken())
    assert llm.requests[0][:3] == llm.requests[1][:3]
    markdown_message = llm.requests[0][2]["content"]
    assert markdown_message.index("HEAD") < markdown_message.index("MIDDLE") < markdown_message.index("TAIL")
    state.close(); knowledge.close()


def test_fixed_prefix_maps_markdown_image_reference_to_tool_media_id() -> None:
    document = {"id": "doc", "title": "Paper"}
    artifact = {"checksum": "abc"}
    media = [{"id": "doc_figure_1", "page": 3, "label": "figure1", "caption": "Architecture", "metadata": {"markdown_reference": "images/figure_1.png"}}]
    identity = DocumentContextBuilder().build_fixed_prefix(document, artifact, "![figure](images/figure_1.png)", media)[1]["content"]
    assert '"markdown_reference":"images/figure_1.png"' in identity
    assert '"media_id":"doc_figure_1"' in identity


def test_long_answer_is_saved_and_preview_is_explicit(tmp_path: Path) -> None:
    _, knowledge, state, _, workspace, config, files, _ = _fixture(tmp_path)
    config.document_long_answer_tokens = 10
    class Estimator:
        def estimate_text(self, text): return len(text)
    preview, file_id, metadata = LongResponseService(config, files, Estimator()).finalize(workspace["id"], "A" * 100)
    assert file_id and metadata["complete"] is False
    assert "MIDDLE OMITTED" in preview and 'complete="false"' in preview
    assert files.read_text(workspace["id"], file_id)["text"] == "A" * 100
    state.close(); knowledge.close()


def test_document_budget_defaults_and_invalid_relationship() -> None:
    config = AutoMemoryConfig()
    assert (config.document_context_window_tokens, config.document_max_input_tokens, config.document_compaction_trigger_tokens, config.document_compaction_target_tokens) == (1_000_000, 920_000, 850_000, 780_000)
    config.document_compaction_target_tokens = 900_000
    with pytest.raises(Exception):
        validate_config(config)
