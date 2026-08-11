"""Single-terminal REPL and one-shot command execution."""

from __future__ import annotations

from contextlib import contextmanager
import signal
import shutil
import time

from .cancellation import CancellationToken
from .errors import AutoMemoryError, CancelledError, ExitCode, UsageError
from .models import CommandResult, InputKind
from . import __version__
from .branding import render_startup_banner
from .provider_presets import match_preset


@contextmanager
def _foreground_cancellation(token: CancellationToken):
    previous = None

    def cancel(_signum, _frame) -> None:
        token.cancel()

    try:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, cancel)
    except (ValueError, AttributeError):
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def dispatch(text: str, ctx, output, router) -> CommandResult:
    parsed = router.parse(text)
    if parsed.kind == InputKind.EMPTY:
        return CommandResult()
    token = CancellationToken()
    with _foreground_cancellation(token):
        if parsed.kind == InputKind.COMMAND:
            spec = router.resolve(parsed.name)
            return spec.handler(ctx, parsed.arguments, output, token, router)
        if parsed.kind == InputKind.RAG_CHAT:
            if not parsed.question:
                raise UsageError("Knowledge query is required", hint="Use /s <question>")
            result = ctx.grounded_chat.stream(ctx.current_conversation, parsed.question, output, token)
            ctx.last_trace = result.get("metadata", {}).get("retrieval_trace", {})
            return CommandResult(data=result)
        result = ctx.direct_chat.stream(ctx.current_conversation, parsed.question, output, token)
        return CommandResult(data=result)


def execute_once(text: str, ctx, output, router, *, debug: bool = False) -> int:
    try:
        result = dispatch(text, ctx, output, router)
        return int(ExitCode.OK if result.ok else ExitCode.INTERNAL)
    except AutoMemoryError as exc:
        if ctx.diagnostics:
            ctx.diagnostics.record_error(exc)
        output.write_error(exc, debug=debug)
        return int(exc.exit_code)
    except KeyboardInterrupt:
        error = CancelledError()
        output.write_error(error, debug=debug)
        return int(error.exit_code)
    except Exception as exc:
        if ctx.diagnostics:
            ctx.diagnostics.record_error(exc)
        output.write_error(exc, debug=debug)
        return int(ExitCode.INTERNAL)


def run_repl(ctx, output, router, *, debug: bool = False) -> int:
    credential_ready = ctx.credentials.configured(ctx.config.llm.credential_name)
    preset = match_preset("llm", ctx.config.llm.base_url)
    provider = preset.label if preset else "Custom OpenAI-compatible"
    summary = f"{provider} / {ctx.config.llm.model}" if credential_ready else "LLM not configured"
    width = shutil.get_terminal_size((80, 24)).columns
    output.stdout.write(render_startup_banner(width=width, color=output.color, version=__version__, llm_summary=summary, needs_setup=not credential_ready))
    output.stdout.flush()
    last_idle_interrupt = 0.0
    while True:
        try:
            line = output.read_line()
        except KeyboardInterrupt:
            now = time.monotonic()
            if now - last_idle_interrupt <= 1.5:
                output.stderr.write("\n")
                return int(ExitCode.OK)
            last_idle_interrupt = now
            output.stderr.write("\nPress Ctrl+C again to exit, or use /exit.\n")
            output.stderr.flush()
            continue
        if line is None:
            output.stdout.write("\n")
            output.stdout.flush()
            return int(ExitCode.OK)
        try:
            result = dispatch(line, ctx, output, router)
            if result.exit_requested:
                return int(ExitCode.OK)
        except CancelledError as exc:
            output.write_error(exc, debug=debug)
        except AutoMemoryError as exc:
            if ctx.diagnostics:
                ctx.diagnostics.record_error(exc)
            output.write_error(exc, debug=debug)
        except Exception as exc:
            if ctx.diagnostics:
                ctx.diagnostics.record_error(exc)
            output.write_error(exc, debug=debug)


def run_pipe(ctx, output, router, *, debug: bool = False) -> int:
    exit_code = int(ExitCode.OK)
    for line in output.stdin:
        try:
            result = dispatch(line.rstrip("\r\n"), ctx, output, router)
            if result.exit_requested:
                break
        except AutoMemoryError as exc:
            output.write_error(exc, debug=debug)
            exit_code = int(exc.exit_code)
            break
        except Exception as exc:
            output.write_error(exc, debug=debug)
            exit_code = int(ExitCode.INTERNAL)
            break
    return exit_code
