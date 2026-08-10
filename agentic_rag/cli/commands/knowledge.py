"""Local knowledge, category, index, trace, and export commands."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from ..errors import UsageError
from ..models import CommandResult, CommandSpec, EventKind, OutputEvent
from ..security import ensure_within, safe_filename
from .utils import pop_flag, pop_option, require_count, resolve_prefix


def _result(output, text: str, data=None) -> CommandResult:
    output.emit(OutputEvent(EventKind.RESULT, text=text, data=data))
    return CommandResult(text=text, data=data)


def add(ctx, args, output, cancel, router):
    args = list(args)
    category = pop_option(args, "--category", "default")
    use_vlm = pop_flag(args, "--vlm")
    require_count(args, 1, "/add <path> [path...] [--category id] [--vlm]")
    results = ctx.ingestion.ingest_local([Path(value) for value in args], str(category), output, cancel, use_vlm=use_vlm)
    ctx.retrieval.rebuild(cancel)
    return _result(output, "\n".join(item["message"] for item in results), results)


def docs(ctx, args, output, cancel, router):
    args = list(args)
    category = pop_option(args, "--category", "all")
    if args:
        raise UsageError("Usage: /docs [--category id]")
    rows = ctx.knowledge.list_documents(str(category))
    text = "\n".join(f"{item['id']}  {item['title']}  {item['source_type']}/{item['parser']}  {item['status']}  chunks={item['chunk_count']} media={item['media_count']}  category={item['category_id']}" for item in rows) or "No documents"
    return _result(output, text, rows)


def doc(ctx, args, output, cancel, router):
    require_count(args, 1, "/doc <document-id>")
    document_id = resolve_prefix(ctx.knowledge.list_documents(), args[0], "Document")
    item = ctx.knowledge.get_document(document_id)
    text = f"{item['id']}\nTitle: {item['title']}\nSource: {item['source']}\nType/parser: {item['source_type']}/{item['parser']}\nStatus: {item['status']}\nPages: {item['page_count']}\nChunks: {len(item['chunks'])}\nMedia: {len(item['media'])}"
    return _result(output, text, item)


def remove(ctx, args, output, cancel, router):
    args = list(args)
    force = pop_flag(args, "--force")
    require_count(args, 1, "/remove <document-id> [--force]")
    document_id = resolve_prefix(ctx.knowledge.list_documents(), args[0], "Document")
    if not force and not output.confirm(f"Delete document {document_id} and its derived indexes?"):
        raise UsageError("Delete cancelled; use --force in non-interactive mode")
    detail = ctx.knowledge.get_document(document_id)
    ctx.knowledge.delete_document(document_id)
    for media in (detail or {}).get("media", []):
        try:
            Path(media["storage_path"]).unlink(missing_ok=True)
        except OSError:
            pass
    ctx.retrieval.rebuild(cancel)
    return _result(output, f"Deleted document: {document_id}")


def category(ctx, args, output, cancel, router):
    action = args.pop(0).lower() if args else "list"
    rows = ctx.knowledge.list_categories()
    if action == "list":
        text = "\n".join(f"{item['id']}  {item['name']}" for item in rows)
    elif action == "add":
        require_count(args, 1, "/category add <name>")
        item = ctx.knowledge.create_category(" ".join(args))
        text = f"Created category: {item['id']}  {item['name']}"
    elif action == "rename":
        require_count(args, 2, "/category rename <id> <name>")
        category_id = resolve_prefix(rows, args.pop(0), "Category")
        ctx.knowledge.rename_category(category_id, " ".join(args))
        text = f"Renamed category: {category_id}"
    elif action == "delete":
        force = pop_flag(args, "--force")
        require_count(args, 1, "/category delete <id> [--force]")
        category_id = resolve_prefix(rows, args[0], "Category")
        if not force and not output.confirm(f"Delete empty category {category_id}?"):
            raise UsageError("Delete cancelled; use --force in non-interactive mode")
        ctx.knowledge.delete_category(category_id)
        text = f"Deleted category: {category_id}"
    else:
        raise UsageError("Usage: /category [list|add|rename|delete] ...")
    return _result(output, text)


def reindex(ctx, args, output, cancel, router):
    count = ctx.retrieval.rebuild(cancel)
    return _result(output, f"Rebuilt keyword index from {count} chunks")


def trace(ctx, args, output, cancel, router):
    import json
    return _result(output, json.dumps(ctx.last_trace or {"message": "No RAG query has run in this process"}, ensure_ascii=False, indent=2))


def export(ctx, args, output, cancel, router):
    require_count(args, 1, "/export <media-id> [filename]")
    media_rows = ctx.knowledge.list_media()
    media_id = resolve_prefix(media_rows, args[0], "Media")
    media = next(item for item in media_rows if item["id"] == media_id)
    source = Path(media["storage_path"])
    filename = safe_filename(args[1] if len(args) > 1 else source.name)
    destination = ensure_within(ctx.paths.exports_dir, ctx.paths.exports_dir / filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".automemory-export-", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _result(output, f"Exported: {destination}")


def register(router) -> None:
    router.register(CommandSpec("add", "Import local documents", "/add <path> [path...] [--category id] [--vlm]", add, group="Knowledge"))
    router.register(CommandSpec("docs", "List knowledge documents", "/docs [--category id]", docs, group="Knowledge"))
    router.register(CommandSpec("doc", "Show document details", "/doc <document-id>", doc, group="Knowledge"))
    router.register(CommandSpec("remove", "Delete a knowledge document", "/remove <document-id> [--force]", remove, group="Knowledge"))
    router.register(CommandSpec("category", "Manage knowledge categories", "/category [list|add|rename|delete] ...", category, group="Knowledge"))
    router.register(CommandSpec("reindex", "Rebuild the derived keyword index", "/reindex", reindex, group="Knowledge"))
    router.register(CommandSpec("trace", "Show the last retrieval trace", "/trace", trace, group="Knowledge"))
    router.register(CommandSpec("export", "Export a media asset", "/export <media-id> [filename]", export, group="Knowledge"))
