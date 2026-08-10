"""Retrieval evaluation command."""

import json
from pathlib import Path

from ..models import CommandResult, CommandSpec, EventKind, OutputEvent
from .utils import pop_flag, pop_option, require_count


def evaluate(ctx, args, output, cancel, router):
    args = list(args)
    mode = str(pop_option(args, "--mode", ctx.config.retrieval_mode))
    scope = str(pop_option(args, "--scope", "all"))
    top_k = int(pop_option(args, "--top-k", str(ctx.config.top_k)))
    export = pop_flag(args, "--export")
    require_count(args, 1, "/eval <dataset.json> [--mode mode] [--top-k N] [--scope id] [--export]")
    result = ctx.evaluation.run(Path(args[0]).resolve(), mode, top_k, scope, output, cancel)
    text = json.dumps(result["summary"], ensure_ascii=False, indent=2)
    if export:
        destination = ctx.evaluation.export(result)
        ctx.state.update_evaluation(result["run_id"], "success", result["summary"], str(destination))
        text += f"\nExported: {destination}"
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text, data=result)


def register(router) -> None:
    router.register(CommandSpec("eval", "Evaluate retrieval from a JSON dataset", "/eval <dataset.json> [--mode mode] [--top-k N] [--scope id] [--export]", evaluate, group="Evaluation"))
