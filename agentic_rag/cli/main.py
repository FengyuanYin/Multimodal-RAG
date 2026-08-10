"""AutoMemory executable entry point."""

from __future__ import annotations

import sys

from . import __version__
from .arguments import parse_args
from .commands import register_all
from .context import AppContext
from .errors import AutoMemoryError, ExitCode
from .paths import AutoMemoryPaths
from .repl import execute_once, run_pipe, run_repl
from .router import CommandRouter
from .terminal import InteractiveTerminal, PlainTerminal


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    options = parse_args(argv)
    if options.version:
        print(f"AutoMemory {__version__}")
        return int(ExitCode.OK)

    context = None
    fallback = PlainTerminal(interactive=False)
    try:
        paths = AutoMemoryPaths.resolve(options.home)
        router = CommandRouter()
        register_all(router)
        context = AppContext.create(paths)

        if options.prompt is not None:
            terminal = PlainTerminal(interactive=False)
            return execute_once(options.prompt, context, terminal, router, debug=options.debug)

        interactive = sys.stdin.isatty() and sys.stdout.isatty() and not options.force_plain
        if interactive:
            terminal = InteractiveTerminal(paths.history_file, router.complete, no_color=options.no_color)
            return run_repl(context, terminal, router, debug=options.debug)

        terminal = PlainTerminal(interactive=False)
        return run_pipe(context, terminal, router, debug=options.debug)
    except AutoMemoryError as exc:
        fallback.write_error(exc, debug=options.debug)
        return int(exc.exit_code)
    except KeyboardInterrupt:
        return int(ExitCode.CANCELLED)
    except Exception as exc:
        fallback.write_error(exc, debug=options.debug)
        return int(ExitCode.INTERNAL)
    finally:
        if context is not None:
            context.close()


if __name__ == "__main__":
    raise SystemExit(main())
