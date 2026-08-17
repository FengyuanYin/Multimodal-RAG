"""Help, version, diagnostics, paths, and exit commands."""

from collections import defaultdict

from .. import __version__
from ..models import CommandResult, CommandSpec, EventKind, OutputEvent


def help_command(ctx, args, output, cancel, router) -> CommandResult:
    if args and args[0].lower() != "all":
        spec = router.resolve(args[0])
        text = f"/{spec.name} — {spec.summary}\nUsage: {spec.usage}"
        output.emit(OutputEvent(EventKind.RESULT, text=text))
        return CommandResult(text=text)
    show_all = bool(args and args[0].lower() == "all")
    groups = defaultdict(list)
    for spec in router.specs():
        if show_all or spec.primary:
            groups[spec.group].append(spec)
    lines = ["AutoMemory commands" + (" (all)" if show_all else ""), ""]
    if not show_all:
        ordered = ["setup","mode","kb","add","docs","context","remove","graph","help","exit"]
        lookup = {item.name:item for item in router.specs()}
        for name in ordered:
            spec = lookup.get(name)
            if spec:
                lines.append(f"  /{spec.name:<12} {spec.summary}")
        lines.insert(10, "  /s <question> Search the current knowledge base")
        lines.extend(["", "Run /help all for advanced and compatibility commands."])
    else:
        for group in sorted(groups):
            lines.append(group + ":")
            for spec in groups[group]:
                lines.append(f"  /{spec.name:<12} {spec.summary}")
            lines.append("")
    lines.extend(["Chat directly by typing a normal message.", "Use /s <question> to search the local knowledge base.", "Press Ctrl+C once to cancel active work; /exit to quit."])
    text = "\n".join(lines)
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text)


def version_command(ctx, args, output, cancel, router) -> CommandResult:
    text = f"AutoMemory {__version__}"
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text)


def exit_command(ctx, args, output, cancel, router) -> CommandResult:
    return CommandResult(exit_requested=True)


def diagnose_command(ctx, args, output, cancel, router) -> CommandResult:
    lines = []
    for item in ctx.diagnostics.report():
        lines.append(f"[{item.status.upper()}] {item.name}: {item.detail}")
    if "--errors" in args:
        lines.extend(["Recent errors:", *[f"  {item}" for item in ctx.diagnostics.recent_errors()]])
    text = "\n".join(lines)
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text)


def path_command(ctx, args, output, cancel, router) -> CommandResult:
    text = f"Home: {ctx.paths.root}\nExports: {ctx.paths.exports_dir}\nLogs: {ctx.paths.logs_dir}"
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text)


def register(router) -> None:
    router.register(CommandSpec("help", "Show commands or command help", "/help [all|command]", help_command, aliases=("?",), group="Main", primary=True))
    router.register(CommandSpec("version", "Show AutoMemory version", "/version", version_command, group="Core"))
    router.register(CommandSpec("diagnose", "Show local and cloud configuration health", "/diagnose [--errors]", diagnose_command, group="Core"))
    router.register(CommandSpec("path", "Show isolated AutoMemory data paths", "/path", path_command, group="Core"))
    router.register(CommandSpec("exit", "Exit AutoMemory safely", "/exit", exit_command, aliases=("quit",), group="Main", primary=True))
