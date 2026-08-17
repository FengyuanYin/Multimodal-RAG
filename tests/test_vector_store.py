from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_rag.memory.vector_store import (
    MilvusVectorStore,
    VectorFilter,
    VectorRecord,
    VectorStoreFactory,
)


class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, name, datatype, **kwargs):
        item = {"name": name, "type": datatype, **kwargs}
        if "dim" in item:
            item["params"] = {"dim": item.pop("dim")}
        self.fields.append(item)


class FakeIndexes:
    def __init__(self):
        self.items = []

    def add_index(self, **kwargs):
        self.items.append(kwargs)


class FakeMilvusClient:
    DataType = SimpleNamespace(VARCHAR="VARCHAR", JSON="JSON", FLOAT_VECTOR="FLOAT_VECTOR")

    def __init__(self):
        self.collections = {}
        self.upserts = []
        self.searches = []
        self.queries = []
        self.deletes = []
        self.dropped = []
        self.search_result = []
        self.query_result = []
        self.closed = False

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_schema(self, **kwargs):
        self.schema_options = kwargs
        return FakeSchema()

    def prepare_index_params(self):
        self.indexes = FakeIndexes()
        return self.indexes

    def create_collection(self, collection_name, schema, index_params, **kwargs):
        self.collections[collection_name] = {"fields": schema.fields}
        self.create_options = kwargs

    def describe_collection(self, collection_name):
        return self.collections[collection_name]

    def list_collections(self):
        return list(self.collections)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return [self.search_result]

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return list(self.query_result)

    def delete(self, **kwargs):
        self.deletes.append(kwargs)

    def drop_collection(self, collection_name, **kwargs):
        self.dropped.append(collection_name)
        self.collections.pop(collection_name, None)

    def get_collection_stats(self, collection_name, **kwargs):
        return {"row_count": sum(len(item["data"]) for item in self.upserts if item["collection_name"] == collection_name)}

    def close(self):
        self.closed = True


def make_store(client=None, **kwargs):
    return MilvusVectorStore(client=client or FakeMilvusClient(), collection_name="test vectors", **kwargs)


def test_add_creates_cosine_collection_and_uses_stable_upsert_key():
    client = FakeMilvusClient()
    store = make_store(client)
    record = VectorRecord("chunk-1", [3.0, 4.0], {"content": "hello"}, document_id="doc-1")

    assert store.add([record]) == 1
    first = client.upserts[0]["data"][0]
    assert client.schema_options == {"auto_id": False, "enable_dynamic_field": False}
    assert client.indexes.items == [{"field_name": "vector", "index_type": "AUTOINDEX", "metric_type": "COSINE"}]
    assert first["vector"] == pytest.approx([0.6, 0.8])

    store.add([record])
    assert client.upserts[1]["data"][0]["pk"] == first["pk"]


def test_add_routes_different_dimensions_to_different_collections():
    client = FakeMilvusClient()
    store = make_store(client)
    store.add([
        VectorRecord("a", [1.0, 0.0]),
        VectorRecord("b", [1.0, 0.0, 0.0]),
    ])
    assert sorted(client.collections) == ["test_vectors_v1_d2", "test_vectors_v1_d3"]
    assert len(client.upserts) == 2


@pytest.mark.parametrize("vector", [[], [0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]])
def test_invalid_vectors_are_rejected_before_network_call(vector):
    client = FakeMilvusClient()
    store = make_store(client)
    with pytest.raises(ValueError):
        store.add([VectorRecord("bad", vector)])
    assert client.upserts == []


def test_search_uses_filter_parameters_and_stable_tie_break():
    client = FakeMilvusClient()
    store = make_store(client)
    store.add([VectorRecord("seed", [1.0, 0.0])])
    client.search_result = [
        {"distance": 0.7, "entity": {"record_id": "b", "payload": {"content": "B"}}},
        {"distance": 0.7, "entity": {"record_id": "a", "payload": {"content": "A"}}},
    ]
    results = store.search(
        [1.0, 0.0],
        filter=VectorFilter(namespace='cli" or true', knowledge_base_id="kb-1"),
    )
    assert [item.id for item in results] == ["a", "b"]
    call = client.searches[0]
    assert call["filter"] == "namespace == {namespace} and knowledge_base_id == {knowledge_base_id}"
    assert call["filter_params"] == {"namespace": 'cli" or true', "knowledge_base_id": "kb-1"}
    assert results[0].content == "A"


def test_existing_ids_and_delete_only_touch_managed_collections():
    client = FakeMilvusClient()
    store = make_store(client)
    store.add([VectorRecord("seed", [1.0, 0.0])])
    client.collections["unrelated"] = {"fields": []}
    client.query_result = [{"record_id": "a"}]
    scope = VectorFilter(namespace="cli", profile_fingerprint="fp")

    assert store.existing_ids(["a", "b"], scope) == {"a"}
    assert {item["collection_name"] for item in client.queries} == {"test_vectors_v1_d2"}
    assert store.delete(filter=VectorFilter(namespace="cli", document_id="doc")) is True
    assert {item["collection_name"] for item in client.deletes} == {"test_vectors_v1_d2"}
    assert store.delete_collection("unrelated") is False


def test_unscoped_delete_and_unknown_filters_are_rejected():
    store = make_store()
    with pytest.raises(ValueError, match="unscoped"):
        store.delete()
    with pytest.raises(ValueError, match="Unsupported"):
        store.search([1.0, 0.0], filter={"raw_expression": "true"})


def test_factory_only_accepts_milvus():
    client = FakeMilvusClient()
    store = VectorStoreFactory.create("milvus", client=client)
    assert isinstance(store, MilvusVectorStore)
    with pytest.raises(ValueError, match="Unsupported"):
        VectorStoreFactory.create("chroma", client=client)
