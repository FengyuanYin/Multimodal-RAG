"""Help, version, diagnostics, paths, and exit commands."""

from collections import defaultdict

from .. import __version__
from ..models import CommandResult, CommandSpec, EventKind, OutputEvent


def help_command(ctx, args, output, cancel, router) -> CommandResult:
    if args:
        spec = router.resolve(args[0])
        text = f"/{spec.name} — {spec.summary}\nUsage: {spec.usage}"
        output.emit(OutputEvent(EventKind.RESULT, text=text))
        return CommandResult(text=text)
    groups = defaultdict(list)
    for spec in router.specs():
        groups[spec.group].append(spec)
    lines = ["AutoMemory commands", ""]
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
    router.register(CommandSpec("help", "Show commands or command help", "/help [command]", help_command, aliases=("?",), group="Core"))
    router.register(CommandSpec("version", "Show AutoMemory version", "/version", version_command, group="Core"))
    router.register(CommandSpec("diagnose", "Show local and cloud configuration health", "/diagnose [--errors]", diagnose_command, group="Core"))
    router.register(CommandSpec("path", "Show isolated AutoMemory data paths", "/path", path_command, group="Core"))
    router.register(CommandSpec("exit", "Exit AutoMemory safely", "/exit", exit_command, aliases=("quit",), group="Core"))
