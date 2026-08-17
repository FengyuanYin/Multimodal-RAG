from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.commands.knowledge import reindex, remove
from agentic_rag.cli.commands.knowledge_bases import kb
from agentic_rag.cli.models import ChunkRecord, DocumentRecord
from agentic_rag.cli.paths import AutoMemoryPaths
from agentic_rag.cli.storage import KnowledgeRepository
from agentic_rag.cli.terminal import PlainTerminal


class FakeVectors:
    def __init__(self):
        self.filters = []
        self.collections = ["automemory_vectors_v1_d2"]
        self.dropped = []

    def delete(self, *, filter):
        self.filters.append(filter)
        return True

    def list_collections(self):
        return list(self.collections)

    def delete_collection(self, name):
        self.dropped.append(name)
        return True


class Workspaces:
    def invalidate_document(self, *_args):
        pass


def output():
    return PlainTerminal(stdout=StringIO(), stderr=StringIO(), interactive=False)


def make_repo(paths):
    return KnowledgeRepository(paths.knowledge_db, paths.backups_dir)


def test_remove_deletes_milvus_scope_before_sqlite_document(tmp_path):
    paths = AutoMemoryPaths.resolve(tmp_path)
    repo = make_repo(paths)
    repo.commit_document(
        DocumentRecord("doc", "fp", "Doc", "doc.txt", "text", "default", "text", 1, "ready"),
        [ChunkRecord("chunk", "doc", 1, 0, "evidence")], [],
    )
    vectors = FakeVectors()
    ctx = SimpleNamespace(
        config=SimpleNamespace(active_category="default"), knowledge=repo, vector_store=vectors,
        state=SimpleNamespace(workspaces=Workspaces()), paths=paths,
        document_artifacts=SimpleNamespace(remove_artifact_files=lambda _item: None),
        retrieval=SimpleNamespace(rebuild=lambda _cancel: 0),
    )

    remove(ctx, ["doc", "--force"], output(), CancellationToken(), None)

    assert vectors.filters[0].namespace == "cli"
    assert vectors.filters[0].document_id == "doc"
    assert repo.get_document("doc") is None
    repo.close()


def test_delete_knowledge_base_deletes_only_its_milvus_scope(tmp_path):
    paths = AutoMemoryPaths.resolve(tmp_path)
    repo = make_repo(paths)
    base = repo.create_knowledge_base("Research")
    repo.commit_document(
        DocumentRecord("doc", "fp", "Doc", "doc.txt", "text", base["id"], "text", 1, "ready"),
        [ChunkRecord("chunk", "doc", 1, 0, "evidence")], [],
    )
    vectors = FakeVectors()
    ctx = SimpleNamespace(
        config=SimpleNamespace(active_category="default"), knowledge=repo, vector_store=vectors,
        state=SimpleNamespace(workspaces=Workspaces()), paths=paths,
        retrieval=SimpleNamespace(rebuild=lambda _cancel: 0), save_config=lambda _config: None,
    )

    kb(ctx, ["delete", base["id"], "--force"], output(), CancellationToken(), None)

    assert vectors.filters[0].namespace == "cli"
    assert vectors.filters[0].knowledge_base_id == base["id"]
    assert all(item["id"] != base["id"] for item in repo.list_knowledge_bases())
    repo.close()


def test_force_reindex_drops_only_managed_collections_and_reports_vectors(tmp_path):
    paths = AutoMemoryPaths.resolve(tmp_path)
    repo = make_repo(paths)
    vectors = FakeVectors()
    indexer = SimpleNamespace(ensure=lambda *_args, **_kwargs: {
        "ready": [{"document_id": "doc", "index": "embedding"}], "degraded": [],
    })
    ctx = SimpleNamespace(
        knowledge=repo, vector_store=vectors, retrieval=SimpleNamespace(rebuild=lambda _cancel: 3),
        index_preparation=indexer,
    )

    result = reindex(ctx, ["--force"], output(), CancellationToken(), None)

    assert vectors.dropped == ["automemory_vectors_v1_d2"]
    assert "vector documents=1" in result.text
    repo.close()
