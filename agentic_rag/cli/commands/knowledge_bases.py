"""Knowledge-base lifecycle commands."""

from __future__ import annotations

from pathlib import Path

from ...memory.vector_store import VectorFilter
from ..errors import UsageError
from ..models import CommandResult, CommandSpec, EventKind, OutputEvent
from .utils import pop_flag, require_count


def _save_active(ctx, category_id: str) -> None:
    payload = ctx.config.to_dict()
    payload["active_category"] = category_id
    ctx.save_config(type(ctx.config).from_dict(payload))


def kb(ctx, args, output, cancel, router):
    args = list(args)
    action = args.pop(0).lower() if args else "list"
    if action in {"list", "ls"}:
        rows = ctx.knowledge.list_knowledge_bases()
        text = "\n".join(
            f"{'*' if item['id'] == ctx.config.active_category else ' '} {item['id']}  {item['name']}  documents={item['document_count']}"
            for item in rows
        )
    elif action == "create":
        require_count(args, 1, "/kb create <name>")
        item = ctx.knowledge.create_knowledge_base(" ".join(args))
        _save_active(ctx, item["id"])
        text = f"Created and selected knowledge base: {item['name']} ({item['id']})"
    elif action == "use":
        require_count(args, 1, "/kb use <id-or-name>")
        category_id = ctx.knowledge.resolve_knowledge_base(" ".join(args))
        _save_active(ctx, category_id)
        item = next(row for row in ctx.knowledge.list_knowledge_bases() if row["id"] == category_id)
        documents = ctx.knowledge.list_documents(category_id)
        text = f"Current knowledge base: {item['name']} ({category_id})"
    elif action == "rename":
        require_count(args, 2, "/kb rename <id-or-name> <new-name>")
        category_id = ctx.knowledge.resolve_knowledge_base(args.pop(0))
        ctx.knowledge.rename_knowledge_base(category_id, " ".join(args))
        text = f"Renamed knowledge base: {category_id}"
    elif action == "delete":
        force = pop_flag(args, "--force")
        require_count(args, 1, "/kb delete <id-or-name> [--force]")
        category_id = ctx.knowledge.resolve_knowledge_base(" ".join(args))
        item = next(row for row in ctx.knowledge.list_knowledge_bases() if row["id"] == category_id)
        documents = ctx.knowledge.list_documents(category_id)
        if item["document_count"] and not force and not output.confirm(f"Delete knowledge base {item['name']} and all its documents?"):
            raise UsageError("Delete cancelled; use --force in non-interactive mode")
        if documents:
            if ctx.vector_store is None:
                raise UsageError("Milvus is unavailable; knowledge-base deletion was not started")
            ctx.vector_store.delete(filter=VectorFilter(namespace="cli", knowledge_base_id=category_id))
        paths = ctx.knowledge.delete_knowledge_base(category_id, force=force or bool(item["document_count"]))
        for value in paths:
            try:
                if str(value).startswith("artifact:"):
                    artifact_path = (ctx.paths.knowledge_assets_dir / str(value)[9:]).resolve()
                    artifact_path.relative_to(ctx.paths.knowledge_assets_dir.resolve())
                    artifact_path.unlink(missing_ok=True)
                    artifact_path.with_suffix(".json").unlink(missing_ok=True)
                else:
                    Path(value).unlink(missing_ok=True)
            except OSError:
                pass
        for document in documents:
            ctx.state.workspaces.invalidate_document(document["id"], "deleted")
        if ctx.config.active_category == category_id:
            _save_active(ctx, "default")
        else:
            ctx.retrieval.rebuild(cancel)
        text = f"Deleted knowledge base: {item['name']}"
    else:
        raise UsageError("Usage: /kb [list|create|use|rename|delete] ...")
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text)


def register(router) -> None:
    router.register(CommandSpec("kb", "Manage knowledge bases", "/kb [list|create|use|rename|delete] ...", kb, group="Main", primary=True))
