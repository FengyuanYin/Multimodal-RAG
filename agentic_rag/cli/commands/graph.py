"""Graph export command."""

from __future__ import annotations

from ..errors import UsageError
from ..models import CommandResult, CommandSpec, EventKind, OutputEvent


def graph(ctx, args, output, cancel, router):
    args = list(args)
    kind = args.pop(0).lower() if args and args[0].lower() in {"entity","reference","combined"} else "combined"
    if len(args) > 1:
        raise UsageError("Usage: /graph [entity|reference|combined] [filename.png]")
    result = ctx.graph_export.export(ctx.config.active_category, kind, args[0] if args else None)
    text = f"Graph exported: {result.png_path}\nMetadata: {result.metadata_path}\nNodes: {result.exported_nodes}/{result.original_nodes}; edges: {result.exported_edges}/{result.original_edges}"
    if result.truncated:
        text += "\nA deterministic readable subgraph was exported."
    output.emit(OutputEvent(EventKind.RESULT, text=text, data=result))
    return CommandResult(text=text, data=result)


def register(router) -> None:
    router.register(CommandSpec("graph", "Export the current knowledge graph", "/graph [entity|reference|combined] [filename.png]", graph, group="Main", primary=True))
