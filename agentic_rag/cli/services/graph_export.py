"""Headless Matplotlib export for persisted AutoMemory graphs."""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import ConfigurationError
from ..models import GraphExportResult
from ..security import safe_filename


class GraphExportService:
    MAX_NODES = 180

    def __init__(self, knowledge, exports_dir: Path) -> None:
        self.knowledge, self.exports_dir = knowledge, exports_dir

    def export(self, category_id: str, kind: str = "combined", filename: str | None = None) -> GraphExportResult:
        if kind not in {"entity", "reference", "combined"}:
            raise ConfigurationError("Graph type must be entity, reference, or combined")
        nodes, edges = self.knowledge.load_graph(category_id, kind)
        if not nodes:
            raise ConfigurationError("The current knowledge base has no graph data", hint="Use /mode advanced, then import or reindex documents")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
            import networkx as nx
        except ImportError as exc:
            raise ConfigurationError("Graph export requires Matplotlib", hint="Reinstall the AutoMemory CLI package") from exc
        original_nodes, original_edges = len(nodes), len(edges)
        degree: dict[tuple[str,str], int] = {}
        for edge in edges:
            degree[(edge["graph_kind"],edge["source_id"])] = degree.get((edge["graph_kind"],edge["source_id"]),0)+1
            degree[(edge["graph_kind"],edge["target_id"])] = degree.get((edge["graph_kind"],edge["target_id"]),0)+1
        ranked = sorted(nodes, key=lambda item:(-bool(item.get("evidence_chunk_id")),-degree.get((item["graph_kind"],item["id"]),0),item["graph_kind"],item["id"]))
        selected = ranked[:self.MAX_NODES]
        selected_ids = {(item["graph_kind"],item["id"]) for item in selected}
        selected_edges = [item for item in edges if (item["graph_kind"],item["source_id"]) in selected_ids and (item["graph_kind"],item["target_id"]) in selected_ids]
        graph = nx.MultiDiGraph()
        for item in selected:
            key = f"{item['graph_kind']}:{item['id']}"
            graph.add_node(key, **item)
        for item in selected_edges:
            graph.add_edge(f"{item['graph_kind']}:{item['source_id']}",f"{item['graph_kind']}:{item['target_id']}",**item)
        base = safe_filename(filename or f"automemory-{category_id}-{kind}.png")
        if not base.lower().endswith(".png"):
            base += ".png"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        png_path = (self.exports_dir / base).resolve()
        try:
            png_path.relative_to(self.exports_dir.resolve())
        except ValueError as exc:
            raise ConfigurationError("Graph export path must stay inside the exports directory") from exc
        json_path = png_path.with_suffix(".json")
        figure = plt.figure(figsize=(16, 10), dpi=140)
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        position = nx.spring_layout(graph, seed=42, k=max(0.25, 2.0 / max(1, len(graph) ** 0.5)))
        colors = {"document":"#6957ff","page":"#38bdf8","chunk":"#94a3b8","image":"#f97316","table":"#eab308","entity":"#22c55e","person":"#22c55e","organization":"#14b8a6","concept":"#84cc16","event":"#ec4899"}
        node_colors = [colors.get(graph.nodes[node].get("node_type"), "#a78bfa") for node in graph.nodes]
        nx.draw_networkx_nodes(graph,position,node_color=node_colors,node_size=520,alpha=.9)
        edge_colors = ["#22c55e" if data.get("graph_kind")=="entity" else "#64748b" for _,_,data in graph.edges(data=True)]
        nx.draw_networkx_edges(graph,position,edge_color=edge_colors,width=.8,alpha=.55,arrows=True,arrowsize=9)
        labels = {node:(str(data.get("label") or node)[:28]+("…" if len(str(data.get("label") or node))>28 else "")) for node,data in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph,position,labels,font_size=6,font_family="sans-serif")
        present_types = sorted({data.get("node_type", "other") for _, data in graph.nodes(data=True)})
        handles = [Patch(facecolor=colors.get(node_type,"#a78bfa"),label=node_type) for node_type in present_types]
        handles.extend([Patch(facecolor="#22c55e",label="entity relation"),Patch(facecolor="#64748b",label="document reference")])
        plt.legend(handles=handles,loc="upper left",fontsize=7,framealpha=.9)
        truncated = original_nodes > len(selected)
        plt.title(f"AutoMemory {kind} graph — {len(selected)}/{original_nodes} nodes, {len(selected_edges)}/{original_edges} edges" + (" (subgraph)" if truncated else ""))
        plt.axis("off"); plt.tight_layout(); figure.savefig(png_path, bbox_inches="tight"); plt.close(figure)
        metadata = {"knowledge_base_id":category_id,"graph_kind":kind,"original_nodes":original_nodes,"original_edges":original_edges,"exported_nodes":len(selected),"exported_edges":len(selected_edges),"truncated":truncated,"nodes":selected,"edges":selected_edges}
        json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return GraphExportResult(png_path,json_path,kind,original_nodes,original_edges,len(selected),len(selected_edges),truncated)
