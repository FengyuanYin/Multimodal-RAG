"""Single-document full-Markdown workspace commands."""

from __future__ import annotations

from ..errors import UsageError
from ..models import CommandResult, CommandSpec, EventKind, OutputEvent
from ..services.document_context import PROMPT_VERSION
from .utils import pop_flag, pop_option, require_count, resolve_prefix


def _emit(output, text: str, data=None):
    output.emit(OutputEvent(EventKind.RESULT, text=text, data=data)); return CommandResult(text=text, data=data)


def context_command(ctx, args, output, cancel, router):
    args = list(args); action = args.pop(0).lower() if args else "status"
    repo = ctx.state.workspaces
    active = repo.active_for_conversation(ctx.current_conversation)
    if action == "open":
        require_count(args, 1, "/context open <document-id>")
        document_id = resolve_prefix(ctx.knowledge.list_documents(ctx.config.active_category), args[0], "Document")
        document = ctx.knowledge.get_document_in_base(document_id, ctx.config.active_category)
        artifact = ctx.knowledge.get_document_artifact(document_id)
        if not document or not artifact: raise UsageError("Document has no complete Markdown; reparse it with /mineru")
        ctx.document_artifacts.verify(artifact)
        model_fp = ctx.llm_client.profile_fingerprint if ctx.llm_client else "not-configured"
        workspace = repo.open_or_create(ctx.current_conversation, document_id, artifact["id"], artifact["checksum"], model_fp, PROMPT_VERSION)
        return _emit(output, f"Full-document workspace active: {document['title']} ({workspace['id']})\nMarkdown: {artifact['byte_size']} bytes | source={artifact['source']} | main model={ctx.config.llm.model}", workspace)
    if action == "status":
        if not active: return _emit(output, "No active full-document workspace")
        document = ctx.knowledge.get_document(active["document_id"])
        return _emit(output, f"Active full-document workspace: {active['id']}\nDocument: {(document or {}).get('title', active['document_id'])}\nStatus: {active['status']}", active)
    if action == "leave":
        repo.set_active(ctx.current_conversation, None); return _emit(output, "Left the full-document workspace; normal messages now use direct chat")
    if not active: raise UsageError("No active full-document workspace", hint="Use /context open <document-id>")
    if action == "clear":
        force = pop_flag(args,"--force")
        if args: raise UsageError("Usage: /context clear [--force]")
        if not force and not output.confirm("Clear this document workspace history?"): raise UsageError("Clear cancelled; use --force in non-interactive mode")
        repo.clear_events(active["id"]); return _emit(output,"Document workspace history cleared; source Markdown was preserved")
    if action == "files":
        rows = repo.list_files(active["id"]); text = "\n".join(f'{item["id"]}  {item["file_kind"]}  {item["display_name"]}  bytes={item["byte_size"]}  status={item["status"]}' for item in rows) or "No generated workspace files"
        return _emit(output,text,rows)
    if action == "read":
        start = int(pop_option(args,"--start","0")); maximum = int(pop_option(args,"--max-chars","48000")); require_count(args,1,"/context read <file-id> [--start N] [--max-chars N]")
        result = ctx.workspace_files.read_text(active["id"],args[0],start,maximum); return _emit(output,result["text"],result)
    if action == "export":
        require_count(args,1,"/context export <file-id> [filename]"); path = ctx.workspace_files.export_file(active["id"],args[0],args[1] if len(args)>1 else None); return _emit(output,f"Exported: {path}")
    if action == "delete-file":
        force = pop_flag(args,"--force"); require_count(args,1,"/context delete-file <file-id> [--force]")
        if not force and not output.confirm(f"Delete generated workspace file {args[0]}?"): raise UsageError("Delete cancelled; use --force in non-interactive mode")
        ctx.workspace_files.delete_generated_file(active["id"],args[0]); return _emit(output,f"Deleted workspace file: {args[0]}")
    raise UsageError("Usage: /context [open|status|leave|clear|files|read|export|delete-file] ...")


def register(router) -> None:
    router.register(CommandSpec("context", "Use one complete MinerU Markdown with the main LLM", "/context [open|status|leave|clear|files|read|export|delete-file] ...", context_command, group="Main", primary=True))
