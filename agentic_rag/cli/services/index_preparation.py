"""Build retryable cloud and graph indexes for CLI knowledge bases."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ...memory.vector_store import VectorFilter, VectorRecord
from ...memory.media_association import associate_references
from ..cancellation import CancellationToken
from ..errors import CancelledError
from ..models import EventKind, GraphEdgeRecord, GraphNodeRecord, OutputEvent, RagPreset


class IndexPreparationService:
    VERSION = "cli-graph-v1"
    VECTOR_VERSION = "cli-milvus-v1"

    def __init__(self, knowledge, *, vector_store=None, embedding_client=None, llm_client=None, batch_delay_seconds: float = 0.0) -> None:
        self.knowledge = knowledge
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.batch_delay_seconds = batch_delay_seconds

    @staticmethod
    def _emit(output, phase: str, text: str, completed: int = 0, total: int = 0) -> None:
        if output:
            output.emit(OutputEvent(EventKind.PROGRESS, text=text, phase=phase, completed=completed, total=total))

    def ensure(self, category_id: str, preset: RagPreset, output, cancel: CancellationToken, *, document_ids: set[str] | None = None) -> dict[str, Any]:
        documents = [item for item in self.knowledge.list_documents(category_id, include_error=False) if not document_ids or item["id"] in document_ids]
        report: dict[str, Any] = {"ready": [], "degraded": []}
        for position, document in enumerate(documents, 1):
            cancel.checkpoint()
            doc_id = document["id"]
            chunks = self.knowledge.list_chunks(doc_id)
            media = self.knowledge.list_media(doc_id)
            states = {(item["index_kind"],item["profile_fingerprint"]): item for item in self.knowledge.index_states(doc_id) if item["status"] == "ready" and item["version"] == self.VERSION}
            if "vector" in preset.channels:
                if not self.embedding_client or not self.vector_store:
                    report["degraded"].append({"document_id": doc_id, "index": "embedding", "reason": "not configured"})
                else:
                    try:
                        fingerprint = self.embedding_client.profile_fingerprint
                        scope = VectorFilter(
                            namespace="cli",
                            document_id=doc_id,
                            knowledge_base_id=document["category_id"],
                            profile_fingerprint=fingerprint,
                        )
                        existing = self.vector_store.existing_ids([item["id"] for item in chunks], scope)
                        missing = [item for item in chunks if item["id"] not in existing]
                        if missing:
                            self._emit(output, "embedding", f"Embedding {document['title']} in the cloud", 0, len(missing))
                            def progress(completed: int, total: int) -> None:
                                self._emit(output, "embedding", f"Embedding {document['title']} in the cloud", completed, total)
                            vectors = self.embedding_client.embeddings([item["text"] for item in missing], cancel, on_progress=progress, batch_delay_seconds=self.batch_delay_seconds)
                            records = [VectorRecord(
                                id=item["id"],
                                vector=vector,
                                namespace="cli",
                                document_id=item["document_id"],
                                knowledge_base_id=item["category_id"],
                                profile_fingerprint=fingerprint,
                                payload={
                                    "content": item["text"],
                                    "text": item["text"],
                                    "document_id": item["document_id"],
                                    "document": item["document"],
                                    "page": int(item["page"]),
                                    "modality": item["modality"],
                                    "media_refs": item.get("media_refs") or [],
                                },
                            ) for item, vector in zip(missing, vectors)]
                            self.vector_store.add(records)
                        self.knowledge.set_index_state(doc_id, "embedding", fingerprint, self.VECTOR_VERSION, "ready")
                        report["ready"].append({"document_id": doc_id, "index": "embedding"})
                    except CancelledError:
                        raise
                    except Exception as exc:
                        self.knowledge.set_index_state(doc_id, "embedding", getattr(self.embedding_client, "profile_fingerprint", ""), self.VERSION, "error", type(exc).__name__)
                        report["degraded"].append({"document_id": doc_id, "index": "embedding", "reason": type(exc).__name__})
            if "reference_graph" in preset.channels or "multimodal" in preset.channels:
                if ("reference_graph", "") not in states:
                    self._emit(output, "reference_graph", f"Building document reference graph: {document['title']}", position - 1, len(documents))
                    nodes, edges = self._reference_graph(document, chunks, media)
                    self.knowledge.replace_document_graph(doc_id, "reference", nodes, edges)
                    self.knowledge.set_index_state(doc_id, "reference_graph", "", self.VERSION, "ready")
                report["ready"].append({"document_id": doc_id, "index": "reference_graph"})
            if "entity_graph" in preset.channels:
                if not self.llm_client:
                    report["degraded"].append({"document_id": doc_id, "index": "entity_graph", "reason": "not configured"})
                else:
                    try:
                        if ("entity_graph", self.llm_client.profile_fingerprint) not in states:
                            self._emit(output, "entity_graph", f"Extracting entity graph in the cloud: {document['title']}", position - 1, len(documents))
                            nodes, edges = self._entity_graph(document, chunks, cancel)
                            self.knowledge.replace_document_graph(doc_id, "entity", nodes, edges)
                            self.knowledge.set_index_state(doc_id, "entity_graph", self.llm_client.profile_fingerprint, self.VERSION, "ready")
                        report["ready"].append({"document_id": doc_id, "index": "entity_graph"})
                    except CancelledError:
                        raise
                    except Exception as exc:
                        self.knowledge.set_index_state(doc_id, "entity_graph", self.llm_client.profile_fingerprint, self.VERSION, "error", type(exc).__name__)
                        report["degraded"].append({"document_id": doc_id, "index": "entity_graph", "reason": type(exc).__name__})
        return report

    @staticmethod
    def _node_id(*parts: str) -> str:
        raw = "\0".join(parts)
        return "g_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _reference_graph(self, document, chunks, media):
        kb, doc_id = document["category_id"], document["id"]
        nodes: dict[str, GraphNodeRecord] = {}
        edges: list[GraphEdgeRecord] = []
        doc_node = f"document:{doc_id}"
        nodes[doc_node] = GraphNodeRecord(doc_node, kb, "reference", "document", document["title"], doc_id)
        media_for_rules = []
        for item in media:
            number = re.search(r"\d{1,3}", str(item.get("label") or ""))
            canonical = f"{'表' if item['media_type'] == 'table' else '图'}{number.group(0)}" if number else item["label"]
            media_for_rules.append({**item, "doc_id": doc_id, "type": item["media_type"], "label": canonical})
        for chunk in chunks:
            page_node, chunk_node = f"page:{doc_id}:{chunk['page']}", f"chunk:{chunk['id']}"
            nodes[page_node] = GraphNodeRecord(page_node, kb, "reference", "page", f"Page {chunk['page']}", doc_id, chunk["page"])
            nodes[chunk_node] = GraphNodeRecord(chunk_node, kb, "reference", "chunk", chunk["text"][:80], doc_id, chunk["page"], chunk["id"])
            for source, target, relation in ((doc_node,page_node,"has_page"),(page_node,chunk_node,"contains")):
                edge_id = self._node_id("reference", source, target, relation)
                edges.append(GraphEdgeRecord(edge_id,kb,"reference",source,target,relation,doc_id,chunk["id"]))
            refs = associate_references(chunk["text"], doc_id, chunk["page"], media_for_rules)
            self.knowledge.update_chunk_media_refs(chunk["id"], refs)
            for ref in refs:
                if not ref["media_id"]:
                    continue
                media_node = f"media:{ref['media_id']}"
                asset = next((item for item in media if item["id"] == ref["media_id"]), None)
                if not asset:
                    continue
                nodes[media_node] = GraphNodeRecord(media_node,kb,"reference",asset["media_type"],asset["label"],doc_id,asset["page"],chunk["id"],{"caption":asset["caption"]})
                edge_id = self._node_id("reference",chunk_node,media_node,"references",str(ref["offset"]))
                props = {**ref,"anchor_page":chunk["page"],"target_page":asset["page"]}
                edges.append(GraphEdgeRecord(edge_id,kb,"reference",chunk_node,media_node,"references",doc_id,chunk["id"],props))
        for asset in media:
            page_node, media_node = f"page:{doc_id}:{asset['page']}", f"media:{asset['id']}"
            nodes.setdefault(page_node, GraphNodeRecord(page_node,kb,"reference","page",f"Page {asset['page']}",doc_id,asset["page"]))
            nodes.setdefault(media_node, GraphNodeRecord(media_node,kb,"reference",asset["media_type"],asset["label"],doc_id,asset["page"],None,{"caption":asset["caption"]}))
            edges.append(GraphEdgeRecord(self._node_id("reference",page_node,media_node,"contains"),kb,"reference",page_node,media_node,"contains",doc_id))
        return list(nodes.values()), edges

    def _entity_graph(self, document, chunks, cancel):
        kb, doc_id = document["category_id"], document["id"]
        nodes: dict[str, GraphNodeRecord] = {}
        edges: dict[str, GraphEdgeRecord] = {}
        system = "Extract important entities and explicit relationships. Return JSON with entities [{name,type}] and relations [{source,target,type}]. Do not infer unsupported facts."
        for chunk in chunks:
            cancel.checkpoint()
            data = self.llm_client.complete_json([{"role":"system","content":system},{"role":"user","content":chunk["text"][:5000]}], cancel)
            entities = data.get("entities") if isinstance(data.get("entities"), list) else []
            relations = data.get("relations") if isinstance(data.get("relations"), list) else []
            by_name: dict[str, str] = {}
            for entity in entities[:50]:
                name = str(entity.get("name") or "").strip()
                if not name:
                    continue
                normalized = re.sub(r"\s+", " ", name).casefold()
                node_id = self._node_id("entity",doc_id,normalized)
                by_name[normalized] = node_id
                nodes[node_id] = GraphNodeRecord(node_id,kb,"entity",str(entity.get("type") or "concept"),name,doc_id,chunk["page"],chunk["id"])
            for relation in relations[:80]:
                source_name, target_name = str(relation.get("source") or "").strip(), str(relation.get("target") or "").strip()
                source = by_name.get(re.sub(r"\s+", " ", source_name).casefold())
                target = by_name.get(re.sub(r"\s+", " ", target_name).casefold())
                if not source or not target:
                    continue
                relation_type = str(relation.get("type") or "related_to")[:80]
                edge_id = self._node_id("entity",source,target,relation_type,chunk["id"])
                edges[edge_id] = GraphEdgeRecord(edge_id,kb,"entity",source,target,relation_type,doc_id,chunk["id"],{"source":source_name,"target":target_name})
        return list(nodes.values()), list(edges.values())
