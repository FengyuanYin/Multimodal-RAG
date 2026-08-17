from __future__ import annotations

import pytest

from agentic_rag.cli.config import AutoMemoryConfig, validate_config
from agentic_rag.cli.errors import ConfigurationError


def test_old_config_uses_safe_embedding_delay_default() -> None:
    config = AutoMemoryConfig.from_dict({"retrieval_mode": "hybrid"})
    assert config.embedding_batch_delay_seconds == 1.0
    assert config.milvus_uri == "http://localhost:19530"
    assert config.milvus_database == "default"
    assert config.milvus_collection == "automemory_vectors"


@pytest.mark.parametrize("value", [-0.1, 30.1])
def test_embedding_delay_must_be_bounded(value: float) -> None:
    config = AutoMemoryConfig(embedding_batch_delay_seconds=value)
    with pytest.raises(ConfigurationError, match="embedding_batch_delay_seconds"):
        validate_config(config)
