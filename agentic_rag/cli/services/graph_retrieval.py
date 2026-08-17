"""Evidence-backed retrieval over the CLI's persisted dual graph."""

from __future__ import annotations

import re
import logging

from ..models import RetrievalHit


class GraphRetrievalService:
    def __init__(self, knowledge) -> None:
        self.knowledge = knowledge

    @staticmethod
    def _terms(query: str) -> set[str]:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        words = {item.strip().casefold() for item in jieba.cut(query) if len(item.strip()) >= 2}
        words.update(item.casefold() for item in re.findall(r"[A-Za-z0-9_]{2,}", query))
        return words

    def search(self, query: str, category_id: str, kind: str, depth: int, limit: int, cancel) -> list[RetrievalHit]:
        nodes, edges = self.knowledge.load_graph(category_id, kind)
        if not nodes:
            return []
        terms = self._terms(query)
        node_map = {item["id"]: item for item in nodes}
        matched = {item["id"] for item in nodes if any(term in item["label"].casefold() for term in terms)}
        frontier, visited = set(matched), set(matched)
        paths: dict[str, list[dict]] = {}
        for _ in range(max(0, depth)):
            next_frontier = set()
            for edge in edges:
                cancel.checkpoint()
                if edge["source_id"] in frontier or edge["target_id"] in frontier:
                    other = edge["target_id"] if edge["source_id"] in frontier else edge["source_id"]
                    next_frontier.add(other)
                    if edge.get("evidence_chunk_id"):
                        paths.setdefault(edge["evidence_chunk_id"], []).append({"relation":edge["relation_type"],"source":node_map.get(edge["source_id"],{}).get("label",""),"target":node_map.get(edge["target_id"],{}).get("label","")})
            frontier = next_frontier - visited
            visited |= frontier
            if not frontier:
                break
        for node_id in visited:
            node = node_map.get(node_id, {})
            if node.get("evidence_chunk_id"):
                paths.setdefault(node["evidence_chunk_id"], [])
        chunks = {item["id"]: item for item in self.knowledge.list_chunks(category_id=category_id)}
        hits = []
        for chunk_id, graph_paths in paths.items():
            item = chunks.get(chunk_id)
            if not item:
                continue
            score = 1.0 + min(1.0, len(graph_paths) / 10)
            hits.append(RetrievalHit(item["id"],item["document_id"],item["document"],item["text"],int(item["page"]),item["modality"],score,{f"{kind}_graph":score},item.get("media_refs") or [],graph_paths=graph_paths))
        return sorted(hits, key=lambda item:(-item.score,item.target_id))[:limit]
