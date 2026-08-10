"""Conversation lifecycle commands."""

from ..errors import UsageError
from ..models import CommandResult, CommandSpec, EventKind, OutputEvent
from .utils import pop_flag, require_count, resolve_prefix


def _emit(output, text: str) -> CommandResult:
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text)


def new(ctx, args, output, cancel, router):
    title = " ".join(args).strip() or "New conversation"
    item = ctx.state.create_conversation(title)
    ctx.current_conversation = item["id"]
    return _emit(output, f"Created and selected {item['id']}: {item['title']}")


def sessions(ctx, args, output, cancel, router):
    rows = ctx.state.list_conversations()
    lines = [f"{'*' if item['id'] == ctx.current_conversation else ' '} {item['id']}  {item['title']}" for item in rows]
    return _emit(output, "\n".join(lines) if lines else "No conversations")


def use(ctx, args, output, cancel, router):
    require_count(args, 1, "/use <conversation-id>")
    conversation_id = resolve_prefix(ctx.state.list_conversations(), args[0], "Conversation")
    ctx.state.set_active_conversation(conversation_id)
    ctx.current_conversation = conversation_id
    return _emit(output, f"Selected conversation: {conversation_id}")


def rename(ctx, args, output, cancel, router):
    require_count(args, 1, "/rename <new title>")
    title = " ".join(args).strip()
    ctx.state.rename_conversation(ctx.current_conversation, title)
    return _emit(output, f"Renamed current conversation to: {title}")


def clear(ctx, args, output, cancel, router):
    force = pop_flag(args, "--force")
    if args:
        raise UsageError("Usage: /clear [--force]")
    if not force and not output.confirm("Clear all messages in the current conversation?"):
        raise UsageError("Clear cancelled; use --force in non-interactive mode")
    ctx.state.clear_conversation(ctx.current_conversation)
    return _emit(output, "Current conversation cleared")


def delete(ctx, args, output, cancel, router):
    force = pop_flag(args, "--force")
    target = args[0] if args else ctx.current_conversation
    conversation_id = resolve_prefix(ctx.state.list_conversations(), target, "Conversation")
    if not force and not output.confirm(f"Delete conversation {conversation_id}?"):
        raise UsageError("Delete cancelled; use --force in non-interactive mode")
    ctx.state.delete_conversation(conversation_id)
    ctx.current_conversation = ctx.state.ensure_active_conversation()
    return _emit(output, f"Deleted conversation: {conversation_id}")


def register(router) -> None:
    router.register(CommandSpec("new", "Create and select a conversation", "/new [title]", new, group="Sessions"))
    router.register(CommandSpec("sessions", "List conversations", "/sessions", sessions, group="Sessions"))
    router.register(CommandSpec("use", "Switch conversation", "/use <conversation-id>", use, group="Sessions"))
    router.register(CommandSpec("rename", "Rename current conversation", "/rename <title>", rename, group="Sessions"))
    router.register(CommandSpec("clear", "Clear current conversation", "/clear [--force]", clear, group="Sessions"))
    router.register(CommandSpec("delete", "Delete a conversation", "/delete [conversation-id] [--force]", delete, group="Sessions"))
