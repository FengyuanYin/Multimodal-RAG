from agentic_rag.config import Settings
from agentic_rag.memory.vector_store import MilvusVectorStore, VectorStoreFactory


class Client:
    def list_collections(self):
        return []


def test_api_settings_default_to_local_milvus():
    settings = Settings(_env_file=None)
    assert settings.vector_db_type == "milvus"
    assert settings.milvus_uri == "http://localhost:19530"
    assert settings.milvus_database == "default"


def test_factory_passes_milvus_connection_settings():
    store = VectorStoreFactory.create(
        "milvus",
        collection_name="sdk_vectors",
        uri="http://127.0.0.1:19530",
        database="custom",
        timeout_seconds=3.0,
        client=Client(),
    )
    assert isinstance(store, MilvusVectorStore)
    assert store.config.uri == "http://127.0.0.1:19530"
    assert store.config.database == "custom"
    assert store.config.collection_prefix == "sdk_vectors"
