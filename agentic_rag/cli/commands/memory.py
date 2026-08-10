"""Long-term memory commands."""

from ..errors import UsageError
from ..models import CommandResult, CommandSpec, EventKind, OutputEvent
from .utils import require_count, resolve_prefix


def memory(ctx, args, output, cancel, router):
    action = args.pop(0).lower() if args else "list"
    rows = ctx.state.list_memories()
    if action == "list":
        text = "\n".join(f"{'on ' if item['enabled'] else 'off'} {item['id']}  {item['content']}" for item in rows) or "No memories"
    elif action == "add":
        require_count(args, 1, "/memory add <content>")
        item = ctx.state.add_memory(" ".join(args))
        text = f"Added memory: {item['id']}"
    elif action in {"enable", "disable"}:
        require_count(args, 1, f"/memory {action} <memory-id>")
        memory_id = resolve_prefix(rows, args[0], "Memory")
        ctx.state.set_memory_enabled(memory_id, action == "enable")
        text = f"Memory {memory_id} {action}d"
    elif action == "delete":
        require_count(args, 1, "/memory delete <memory-id>")
        memory_id = resolve_prefix(rows, args[0], "Memory")
        ctx.state.delete_memory(memory_id)
        text = f"Deleted memory: {memory_id}"
    else:
        raise UsageError("Usage: /memory [list|add|enable|disable|delete] ...")
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text)


def register(router) -> None:
    router.register(CommandSpec("memory", "Manage enabled long-term memories", "/memory [list|add|enable|disable|delete] ...", memory, group="Memory"))
