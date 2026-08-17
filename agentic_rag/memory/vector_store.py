"""Milvus-backed vector storage shared by the API/SDK and AutoMemory CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Callable, Iterable, List, Optional


SCHEMA_VERSION = 1
DEFAULT_NAMESPACE = "api"
DEFAULT_PROFILE = "default"
_COLLECTION_RE = re.compile(r"[^A-Za-z0-9_]")
_MAX_FILTER_IDS = 512


class VectorStoreError(RuntimeError):
    """Raised when vector storage cannot safely complete an operation."""


@dataclass(frozen=True)
class MilvusConnectionConfig:
    uri: str = "http://localhost:19530"
    database: str = "default"
    token: Optional[str] = None
    collection_prefix: str = "agentic_rag_vectors"
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class VectorFilter:
    namespace: Optional[str] = None
    document_id: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    profile_fingerprint: Optional[str] = None
    record_ids: Optional[List[str]] = None


@dataclass
class VectorRecord:
    id: str
    vector: List[float]
    payload: dict = field(default_factory=dict)
    namespace: str = DEFAULT_NAMESPACE
    document_id: str = ""
    knowledge_base_id: str = ""
    profile_fingerprint: str = DEFAULT_PROFILE


@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict = field(default_factory=dict)
    content: str = ""


class BaseVectorStore:
    def add(self, records: List[VectorRecord]) -> int:
        raise NotImplementedError

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[VectorFilter | dict] = None,
    ) -> List[SearchResult]:
        raise NotImplementedError

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[VectorFilter | dict] = None,
    ) -> bool:
        raise NotImplementedError

    def delete_collection(self, name: str) -> bool:
        raise NotImplementedError

    def count(self, filter: Optional[VectorFilter | dict] = None, dimension: Optional[int] = None) -> int:
        raise NotImplementedError

    def list_collections(self) -> List[str]:
        raise NotImplementedError


class MilvusVectorStore(BaseVectorStore):
    """A dimension-routed Milvus vector store with structured filtering only."""

    _STRING_LIMITS = {
        "pk": 64,
        "record_id": 2048,
        "namespace": 256,
        "document_id": 2048,
        "knowledge_base_id": 512,
        "profile_fingerprint": 256,
    }
    _OUTPUT_FIELDS = [
        "record_id",
        "namespace",
        "document_id",
        "knowledge_base_id",
        "profile_fingerprint",
        "payload",
    ]

    def __init__(
        self,
        collection_name: str = "agentic_rag_vectors",
        embedding_dim: Optional[int] = None,
        uri: str = "http://localhost:19530",
        database: str = "default",
        token: Optional[str] = None,
        timeout_seconds: float = 10.0,
        *,
        client: Any = None,
        client_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = MilvusConnectionConfig(
            uri=uri,
            database=database,
            token=token or None,
            collection_prefix=self._normalize_collection_prefix(collection_name),
            timeout_seconds=float(timeout_seconds),
        )
        if not 0.1 <= self.config.timeout_seconds <= 600:
            raise ValueError("Milvus timeout_seconds must be between 0.1 and 600")
        self.embedding_dim = int(embedding_dim) if embedding_dim else None
        self._client = client or self._create_client(client_factory)
        self._validated_collections: set[str] = set()

    @staticmethod
    def _normalize_collection_prefix(value: str) -> str:
        raw = str(value or "").strip()
        normalized = _COLLECTION_RE.sub("_", raw)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        if not normalized:
            raise ValueError("Milvus collection prefix must contain a letter or number")
        if normalized[0].isdigit():
            normalized = "v_" + normalized
        if len(normalized) > 220:
            raise ValueError("Milvus collection prefix is too long")
        return normalized

    def _create_client(self, client_factory: Optional[Callable[..., Any]]) -> Any:
        if client_factory is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise VectorStoreError("pymilvus is required for Milvus vector storage") from exc
            client_factory = MilvusClient
        kwargs: dict[str, Any] = {"uri": self.config.uri, "db_name": self.config.database}
        if self.config.token:
            kwargs["token"] = self.config.token
        try:
            return client_factory(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"Unable to connect to Milvus at {self.config.uri}") from exc

    def _collection_name(self, dimension: int) -> str:
        if dimension <= 0:
            raise ValueError("Vector dimension must be positive")
        return f"{self.config.collection_prefix}_v{SCHEMA_VERSION}_d{dimension}"

    @staticmethod
    def _stable_pk(record: VectorRecord) -> str:
        raw = "\0".join((record.namespace, record.profile_fingerprint, record.knowledge_base_id, record.id))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalized_vector(vector: Iterable[float]) -> List[float]:
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError("Vector values must be numeric") from exc
        if not values:
            raise ValueError("Vector must not be empty")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Vector values must be finite")
        norm = math.sqrt(sum(value * value for value in values))
        if not norm:
            raise ValueError("Vector must not have zero norm")
        return [value / norm for value in values]

    @classmethod
    def _validate_record(cls, record: VectorRecord) -> None:
        if not isinstance(record.payload, dict):
            raise ValueError("Vector payload must be a dictionary")
        try:
            json.dumps(record.payload, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Vector payload must be valid JSON") from exc
        values = {
            "record_id": record.id,
            "namespace": record.namespace,
            "document_id": record.document_id,
            "knowledge_base_id": record.knowledge_base_id,
            "profile_fingerprint": record.profile_fingerprint,
        }
        for field_name, value in values.items():
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")
            if field_name in {"record_id", "namespace", "profile_fingerprint"} and not value:
                raise ValueError(f"{field_name} must not be empty")
            if len(value) > cls._STRING_LIMITS[field_name]:
                raise ValueError(f"{field_name} exceeds Milvus field length")

    def _ensure_collection(self, dimension: int) -> str:
        name = self._collection_name(dimension)
        if name in self._validated_collections:
            return name
        try:
            if not self._client.has_collection(collection_name=name):
                self._create_collection(name, dimension)
            else:
                self._validate_collection_schema(name, dimension)
            self._validated_collections.add(name)
            return name
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Milvus collection {name} is unavailable") from exc

    def _create_collection(self, name: str, dimension: int) -> None:
        try:
            from pymilvus import DataType
        except ImportError:
            DataType = getattr(self._client, "DataType", None)
            if DataType is None:
                raise VectorStoreError("pymilvus DataType is unavailable")
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=self._STRING_LIMITS["pk"])
        schema.add_field("record_id", DataType.VARCHAR, max_length=self._STRING_LIMITS["record_id"])
        schema.add_field("namespace", DataType.VARCHAR, max_length=self._STRING_LIMITS["namespace"])
        schema.add_field("document_id", DataType.VARCHAR, max_length=self._STRING_LIMITS["document_id"])
        schema.add_field("knowledge_base_id", DataType.VARCHAR, max_length=self._STRING_LIMITS["knowledge_base_id"])
        schema.add_field("profile_fingerprint", DataType.VARCHAR, max_length=self._STRING_LIMITS["profile_fingerprint"])
        schema.add_field("payload", DataType.JSON)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
            timeout=self.config.timeout_seconds,
        )

    def _validate_collection_schema(self, name: str, dimension: int) -> None:
        description = self._client.describe_collection(collection_name=name)
        fields = {item.get("name") or item.get("field_name"): item for item in description.get("fields", [])}
        required = set(self._STRING_LIMITS) | {"payload", "vector"}
        missing = sorted(required - set(fields))
        expected_types = {
            **{field_name: "VARCHAR" for field_name in self._STRING_LIMITS},
            "payload": "JSON",
            "vector": "FLOAT_VECTOR",
        }
        wrong_types = []
        try:
            from pymilvus import DataType
        except ImportError:
            DataType = getattr(self._client, "DataType", None)
        for field_name, expected_name in expected_types.items():
            if field_name not in fields or DataType is None:
                continue
            actual = fields[field_name].get("type") or fields[field_name].get("data_type")
            expected = getattr(DataType, expected_name, expected_name)
            if actual != expected and str(actual).upper().split(".")[-1] != expected_name:
                wrong_types.append(f"{field_name}={actual}")
        vector = fields.get("vector") or {}
        params = vector.get("params") or vector.get("type_params") or {}
        actual_dim = params.get("dim") or vector.get("dim")
        if missing or wrong_types or (actual_dim is not None and int(actual_dim) != dimension):
            if missing:
                detail = f"missing fields {missing}"
            elif wrong_types:
                detail = f"wrong field types {wrong_types}"
            else:
                detail = f"dimension {actual_dim} != {dimension}"
            raise VectorStoreError(f"Milvus collection {name} has incompatible schema: {detail}")
        list_indexes = getattr(self._client, "list_indexes", None)
        describe_index = getattr(self._client, "describe_index", None)
        if callable(list_indexes) and callable(describe_index):
            index_names = list_indexes(collection_name=name)
            details = [describe_index(collection_name=name, index_name=index_name) for index_name in index_names]
            vector_indexes = [item for item in details if item.get("field_name") == "vector"]
            if not vector_indexes or str(vector_indexes[0].get("metric_type", "")).upper() != "COSINE":
                raise VectorStoreError(f"Milvus collection {name} has incompatible vector index")

    @staticmethod
    def _coerce_filter(value: Optional[VectorFilter | dict]) -> VectorFilter:
        if value is None:
            return VectorFilter()
        if isinstance(value, VectorFilter):
            return value
        if isinstance(value, dict):
            allowed = set(VectorFilter.__dataclass_fields__)
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"Unsupported vector filter fields: {sorted(unknown)}")
            return VectorFilter(**value)
        raise TypeError("filter must be VectorFilter, dict, or None")

    @classmethod
    def _compile_filter(cls, value: Optional[VectorFilter | dict]) -> tuple[str, dict[str, Any]]:
        filter_value = cls._coerce_filter(value)
        clauses: List[str] = []
        params: dict[str, Any] = {}
        for field_name in ("namespace", "document_id", "knowledge_base_id", "profile_fingerprint"):
            field_value = getattr(filter_value, field_name)
            if field_value is not None:
                if not isinstance(field_value, str):
                    raise ValueError(f"{field_name} filter must be a string")
                clauses.append(f"{field_name} == {{{field_name}}}")
                params[field_name] = field_value
        if filter_value.record_ids is not None:
            if len(filter_value.record_ids) > _MAX_FILTER_IDS:
                raise ValueError(f"record_ids filter supports at most {_MAX_FILTER_IDS} values per call")
            if not all(isinstance(item, str) for item in filter_value.record_ids):
                raise ValueError("record_ids filter values must be strings")
            clauses.append("record_id in {record_ids}")
            params["record_ids"] = list(filter_value.record_ids)
        return " and ".join(clauses), params

    def add(self, records: List[VectorRecord]) -> int:
        if not records:
            return 0
        grouped: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            self._validate_record(record)
            vector = self._normalized_vector(record.vector)
            if self.embedding_dim and len(vector) != self.embedding_dim:
                raise ValueError(f"Vector dimension {len(vector)} does not match expected {self.embedding_dim}")
            grouped.setdefault(len(vector), []).append({
                "pk": self._stable_pk(record), "record_id": record.id,
                "namespace": record.namespace, "document_id": record.document_id,
                "knowledge_base_id": record.knowledge_base_id,
                "profile_fingerprint": record.profile_fingerprint,
                "payload": record.payload, "vector": vector,
            })
        try:
            for dimension, data in grouped.items():
                self._client.upsert(
                    collection_name=self._ensure_collection(dimension), data=data,
                    timeout=self.config.timeout_seconds,
                )
            return len(records)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Milvus vector upsert failed") from exc

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[VectorFilter | dict] = None,
    ) -> List[SearchResult]:
        vector = self._normalized_vector(query_vector)
        if top_k <= 0:
            return []
        expression, params = self._compile_filter(filter)
        name = self._collection_name(len(vector))
        if not self._client.has_collection(collection_name=name):
            return []
        self._validate_collection_schema(name, len(vector))
        try:
            rows = self._client.search(
                collection_name=name, data=[vector], anns_field="vector",
                filter=expression, filter_params=params, limit=min(int(top_k), 16384),
                output_fields=self._OUTPUT_FIELDS,
                search_params={"metric_type": "COSINE", "params": {}},
                consistency_level="Strong", timeout=self.config.timeout_seconds,
            )
        except Exception as exc:
            raise VectorStoreError(f"Milvus search failed for collection {name}") from exc
        hits = rows[0] if rows else []
        output: List[SearchResult] = []
        for hit in hits:
            entity = hit.get("entity") or {}
            payload = entity.get("payload") or {}
            record_id = entity.get("record_id") or hit.get("record_id") or str(hit.get("id", ""))
            score = hit.get("distance", hit.get("score", 0.0))
            output.append(SearchResult(
                id=str(record_id), score=float(score), payload=payload,
                content=str(payload.get("content") or payload.get("text") or ""),
            ))
        return sorted(output, key=lambda item: (-item.score, item.id))[:top_k]

    def existing_ids(self, record_ids: List[str], filter: Optional[VectorFilter | dict] = None) -> set[str]:
        if not record_ids:
            return set()
        base = self._coerce_filter(filter)
        found: set[str] = set()
        for start in range(0, len(record_ids), _MAX_FILTER_IDS):
            batch = record_ids[start:start + _MAX_FILTER_IDS]
            scoped = VectorFilter(
                namespace=base.namespace, document_id=base.document_id,
                knowledge_base_id=base.knowledge_base_id,
                profile_fingerprint=base.profile_fingerprint, record_ids=batch,
            )
            expression, params = self._compile_filter(scoped)
            for name in self.list_collections():
                rows = self._client.query(
                    collection_name=name, filter=expression, filter_params=params,
                    output_fields=["record_id"], limit=len(batch),
                    consistency_level="Strong", timeout=self.config.timeout_seconds,
                )
                found.update(str(item["record_id"]) for item in rows if item.get("record_id") is not None)
        return found

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[VectorFilter | dict] = None,
    ) -> bool:
        base = self._coerce_filter(filter)
        if ids is not None:
            if base.record_ids is not None:
                raise ValueError("Specify record IDs either in ids or filter, not both")
            base = VectorFilter(
                namespace=base.namespace, document_id=base.document_id,
                knowledge_base_id=base.knowledge_base_id,
                profile_fingerprint=base.profile_fingerprint, record_ids=list(ids),
            )
        expression, params = self._compile_filter(base)
        if not expression:
            raise ValueError("Refusing unscoped Milvus delete")
        try:
            for name in self.list_collections():
                self._client.delete(
                    collection_name=name, filter=expression, filter_params=params,
                    timeout=self.config.timeout_seconds,
                )
            return True
        except Exception as exc:
            raise VectorStoreError("Milvus scoped delete failed") from exc

    def list_collections(self) -> List[str]:
        try:
            prefix = f"{self.config.collection_prefix}_v{SCHEMA_VERSION}_d"
            return sorted(str(name) for name in self._client.list_collections() if str(name).startswith(prefix))
        except Exception as exc:
            raise VectorStoreError("Unable to list Milvus collections") from exc

    def delete_collection(self, name: str) -> bool:
        if name not in self.list_collections():
            return False
        try:
            self._client.drop_collection(collection_name=name, timeout=self.config.timeout_seconds)
            self._validated_collections.discard(name)
            return True
        except Exception as exc:
            raise VectorStoreError(f"Unable to delete Milvus collection {name}") from exc

    def count(self, filter: Optional[VectorFilter | dict] = None, dimension: Optional[int] = None) -> int:
        names = [self._collection_name(dimension)] if dimension else self.list_collections()
        expression, params = self._compile_filter(filter)
        total = 0
        for name in names:
            if not self._client.has_collection(collection_name=name):
                continue
            if expression:
                rows = self._client.query(
                    collection_name=name, filter=expression, filter_params=params,
                    output_fields=["count(*)"], consistency_level="Strong",
                    timeout=self.config.timeout_seconds,
                )
                if rows:
                    total += int(rows[0].get("count(*)", rows[0].get("count", 0)))
            else:
                stats = self._client.get_collection_stats(
                    collection_name=name, timeout=self.config.timeout_seconds,
                )
                total += int(stats.get("row_count", 0))
        return total

    def validate(self) -> dict[str, Any]:
        collections = self.list_collections()
        for name in collections:
            match = re.search(r"_d(\d+)$", name)
            if match:
                self._validate_collection_schema(name, int(match.group(1)))
        return {"status": "ok", "collections": collections, "collection_count": len(collections)}

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


MilvusStore = MilvusVectorStore


class VectorStoreFactory:
    @staticmethod
    def create(
        db_type: str = "milvus",
        collection_name: str = "agentic_rag_vectors",
        embedding_dim: Optional[int] = None,
        uri: str = "http://localhost:19530",
        database: str = "default",
        token: Optional[str] = None,
        timeout_seconds: float = 10.0,
        **kwargs: Any,
    ) -> BaseVectorStore:
        if db_type.lower() != "milvus":
            raise ValueError(f"Unsupported vector database: {db_type}; expected milvus")
        return MilvusVectorStore(
            collection_name=collection_name, embedding_dim=embedding_dim, uri=uri,
            database=database, token=token, timeout_seconds=timeout_seconds, **kwargs,
        )
