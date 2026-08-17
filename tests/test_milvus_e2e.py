"""Opt-in integration test for a real local Milvus instance."""

from __future__ import annotations

import os
import uuid

import pytest

from agentic_rag.memory.vector_store import MilvusVectorStore, VectorFilter, VectorRecord


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MILVUS_INTEGRATION") != "1",
    reason="set RUN_MILVUS_INTEGRATION=1 to test the local Milvus instance",
)


def test_local_milvus_upsert_scope_search_delete_and_cleanup():
    prefix = "codex_agenticrag_test_" + uuid.uuid4().hex[:10]
    store = MilvusVectorStore(collection_name=prefix, uri="http://localhost:19530", database="default")
    try:
        records = [
            VectorRecord(
                "a", [1.0, 0.0], {"content": "alpha", "document_id": "doc-a"},
                namespace="cli", document_id="doc-a", knowledge_base_id="kb-a", profile_fingerprint="fp",
            ),
            VectorRecord(
                "b", [0.0, 1.0], {"content": "beta", "document_id": "doc-b"},
                namespace="cli", document_id="doc-b", knowledge_base_id="kb-b", profile_fingerprint="fp",
            ),
        ]
        assert store.add(records) == 2
        assert store.add(records[:1]) == 1
        assert store.count(VectorFilter(namespace="cli"), dimension=2) == 2
        assert store.existing_ids(
            ["a", "missing"], VectorFilter(namespace="cli", profile_fingerprint="fp")
        ) == {"a"}

        hits = store.search(
            [1.0, 0.0], top_k=5,
            filter=VectorFilter(namespace="cli", knowledge_base_id="kb-a", profile_fingerprint="fp"),
        )
        assert [item.id for item in hits] == ["a"]
        assert hits[0].content == "alpha"

        assert store.delete(filter=VectorFilter(namespace="cli", document_id="doc-a"))
        assert store.count(VectorFilter(namespace="cli"), dimension=2) == 1
        remaining = store.search(
            [0.0, 1.0], top_k=5,
            filter=VectorFilter(namespace="cli", knowledge_base_id="kb-b", profile_fingerprint="fp"),
        )
        assert [item.id for item in remaining] == ["b"]
    finally:
        for collection in store.list_collections():
            store.delete_collection(collection)
        store.close()
