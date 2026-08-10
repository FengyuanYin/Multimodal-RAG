"""Internet search, numbered result preview, and capture commands."""

from ..errors import UsageError
from ..models import CommandResult, CommandSpec, EventKind, OutputEvent
from .utils import pop_flag, pop_option, require_count


def search(ctx, args, output, cancel, router):
    require_count(args, 1, "/search <keywords>")
    query = " ".join(args)
    key = ctx.credentials.get("tavily_api_key")
    ctx.search_results = ctx.web_client.search(query, ctx.config.web_provider, key, cancel)
    lines = [f"[{index}] {item.title}\n    {item.url}\n    {item.snippet}" for index, item in enumerate(ctx.search_results, 1)]
    text = "\n".join(lines) if lines else "No search results"
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text, data=ctx.search_results)


def fetch(ctx, args, output, cancel, router):
    args = list(args)
    category = str(pop_option(args, "--category", "default"))
    yes = pop_flag(args, "--yes")
    require_count(args, 1, "/fetch <result-number|url> [--category id] [--yes]")
    target = args[0]
    if target.isdigit():
        index = int(target)
        if index < 1 or index > len(ctx.search_results):
            raise UsageError("Search result number is not available; run /search first")
        target = ctx.search_results[index - 1].url
    page = ctx.web_client.fetch(target, cancel)
    preview = page.text[:1000] + ("…" if len(page.text) > 1000 else "")
    output.emit(OutputEvent(EventKind.RESULT, text=f"Title: {page.title}\nURL: {page.url}\nCharacters: {len(page.text)}\n\n{preview}"))
    if not yes and not output.confirm("Import this page into the knowledge base?"):
        return CommandResult(text="Preview only; page was not imported")
    result = ctx.ingestion.ingest_web(page, category, output, cancel)
    ctx.retrieval.rebuild(cancel)
    output.emit(OutputEvent(EventKind.RESULT, text=result["message"]))
    return CommandResult(text=result["message"], data=result)


def register(router) -> None:
    router.register(CommandSpec("search", "Search the internet", "/search <keywords>", search, group="Web"))
    router.register(CommandSpec("fetch", "Preview and import a URL or search result", "/fetch <number|url> [--category id] [--yes]", fetch, group="Web"))
