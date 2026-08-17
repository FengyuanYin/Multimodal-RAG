"""Simple product-level RAG mode selection."""

from __future__ import annotations

from ..models import CommandResult, CommandSpec, EventKind, OutputEvent
from ..rag_presets import get_preset, list_presets
from .utils import require_count


def mode(ctx, args, output, cancel, router):
    if len(args) > 1:
        require_count([], 1, "/mode [fast|balanced|multimodal|advanced]")
    if args:
        preset = get_preset(args[0])
        payload = ctx.config.to_dict()
        payload["rag_mode"] = preset.name
        ctx.save_config(type(ctx.config).from_dict(payload))
        report = ctx.index_preparation.ensure(ctx.config.active_category, preset, output, cancel)
        degraded = len(report.get("degraded", []))
        suffix = f"; degraded indexes={degraded}" if degraded else ""
        text = f"RAG mode changed to {preset.name}: {preset.description}{suffix}"
    else:
        lines = [f"Current RAG mode: {ctx.config.rag_mode}", ""]
        for item in list_presets():
            marker = "*" if item.name == ctx.config.rag_mode else " "
            lines.append(f"{marker} {item.name:<10} {item.description}")
        lines.append("\nExample: /mode advanced")
        text = "\n".join(lines)
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text)


def register(router) -> None:
    router.register(CommandSpec("mode", "Choose a fixed RAG mode", "/mode [fast|balanced|multimodal|advanced]", mode, group="Main", primary=True))
