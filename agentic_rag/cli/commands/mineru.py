"""Cloud MinerU PDF parsing command."""

from pathlib import Path

from ..errors import CancelledError
from ..models import CommandResult, CommandSpec, EventKind, OutputEvent
from .utils import pop_flag, pop_option, require_count


def mineru(ctx, args, output, cancel, router):
    args = list(args)
    category = str(pop_option(args, "--category", "default"))
    selfhost = pop_flag(args, "--selfhost")
    require_count(args, 1, "/mineru <pdf-path> [--category id] [--selfhost]")
    path = Path(args[0]).expanduser().resolve()
    client = ctx.mineru_client("selfhost" if selfhost else ctx.config.mineru_mode)
    task_id = ctx.state.create_task("mineru", {"source": str(path)})
    try:
        parsed = client.parse_selfhosted(path, output, cancel, task_id) if selfhost or ctx.config.mineru_mode == "selfhost" else client.parse_official(path, output, cancel, task_id)
        result = ctx.ingestion.ingest_parsed(parsed, str(path), category, output, cancel, "pdf")
        ctx.state.update_task(task_id, "success", "ready", result)
        ctx.retrieval.rebuild(cancel)
    except CancelledError:
        ctx.state.update_task(task_id, "cancelled", "cancelled", {})
        raise
    except Exception as exc:
        ctx.state.update_task(task_id, "error", "failed", {"error": type(exc).__name__})
        raise
    finally:
        client.close()
    output.emit(OutputEvent(EventKind.RESULT, text=result["message"]))
    return CommandResult(text=result["message"], data=result)


def register(router) -> None:
    router.register(CommandSpec("mineru", "Parse a PDF with cloud MinerU", "/mineru <pdf-path> [--category id] [--selfhost]", mineru, group="Knowledge"))
