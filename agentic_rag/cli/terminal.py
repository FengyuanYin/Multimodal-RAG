"""Line-oriented interactive and script-safe terminal output."""

from __future__ import annotations

import getpass
import os
import sys
from typing import TextIO

from .errors import AutoMemoryError, UsageError
from .models import EventKind, OutputEvent


class PlainTerminal:
    def __init__(self, *, stdin: TextIO | None = None, stdout: TextIO | None = None, stderr: TextIO | None = None, interactive: bool = False, color: bool = False) -> None:
        self.stdin, self.stdout, self.stderr = stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr
        ansi_capable = sys.platform != "win32" or bool(os.getenv("WT_SESSION") or os.getenv("ANSICON") or os.getenv("TERM"))
        self.interactive, self.color = interactive, color and ansi_capable and os.getenv("TERM", "").lower() != "dumb" and not bool(os.getenv("NO_COLOR"))
        self._delta_open = False
        self._progress_open = False
        self._progress_width = 0

    def _close_progress(self) -> None:
        if self._progress_open:
            self.stderr.write("\n")
            self.stderr.flush()
            self._progress_open = False
            self._progress_width = 0

    def emit(self, event: OutputEvent) -> None:
        if event.kind == EventKind.DELTA:
            self._close_progress()
            self.stdout.write(event.text)
            self.stdout.flush()
            self._delta_open = True
            return
        if self._delta_open:
            self.stdout.write("\n")
            self.stdout.flush()
            self._delta_open = False
        if event.kind == EventKind.PROGRESS:
            if self.interactive:
                if event.total:
                    completed = max(0, min(int(event.completed), int(event.total)))
                    percent = int(completed * 100 / event.total)
                    filled = min(24, int(percent * 24 / 100))
                    bar = "#" * filled + "-" * (24 - filled)
                    line = f"[{event.phase or 'working'}] [{bar}] {completed}/{event.total} {percent:3d}%"
                    padded = line.ljust(max(self._progress_width, len(line)))
                    self.stderr.write("\r" + padded)
                    self._progress_width = max(self._progress_width, len(line))
                    self._progress_open = completed < event.total
                    if not self._progress_open:
                        self.stderr.write("\n")
                        self._progress_width = 0
                else:
                    self._close_progress()
                    self.stderr.write(f"[{event.phase or 'working'}] {event.text}\n")
                self.stderr.flush()
        elif event.kind == EventKind.SOURCES:
            self.write_sources(event.data or [])
        elif event.kind in {EventKind.WARNING, EventKind.ERROR}:
            self._close_progress()
            self.stderr.write(f"[{event.code or event.kind.value}] {event.text}\n")
            self.stderr.flush()
        elif event.text:
            self._close_progress()
            self.stdout.write(event.text + ("" if event.text.endswith("\n") else "\n"))
            self.stdout.flush()

    def write_sources(self, sources: list[dict]) -> None:
        if not sources:
            return
        self._close_progress()
        self.stdout.write("\nSources:\n")
        for source in sources:
            self.stdout.write(f"  [{source.get('index','?')}] {source.get('document','document')} | page {source.get('page',1)} | {source.get('modality','text')} | score {float(source.get('score',0)):.4f}\n")
        self.stdout.flush()

    def write_error(self, error: Exception, *, debug: bool = False) -> None:
        self._close_progress()
        if self._delta_open:
            self.stdout.write("\n")
            self._delta_open = False
        if isinstance(error, AutoMemoryError):
            self.stderr.write(f"[{error.code}] {error.message}\n")
            if error.hint:
                self.stderr.write(f"Hint: {error.hint}\n")
        else:
            self.stderr.write(f"[INTERNAL_ERROR] {type(error).__name__}: {error}\n" if debug else "[INTERNAL_ERROR] AutoMemory encountered an unexpected error. Run with --debug for details.\n")
        self.stderr.flush()

    def read_line(self, prompt: str = "AutoMemory> ") -> str | None:
        if self.interactive:
            self.stdout.write(prompt)
            self.stdout.flush()
        value = self.stdin.readline()
        return None if value == "" else value.rstrip("\r\n")

    def confirm(self, message: str, default: bool = False) -> bool:
        if not self.interactive:
            return False
        suffix = " [Y/n] " if default else " [y/N] "
        value = self.read_line(message + suffix)
        if value is None or not value.strip():
            return default
        return value.strip().lower() in {"y", "yes"}

    def read_secret(self, prompt: str) -> str:
        if not self.interactive:
            raise UsageError("Secret input requires an interactive terminal", hint="Set the matching AUTOMEMORY_* environment variable for non-interactive use")
        return getpass.getpass(prompt, stream=self.stderr)

    def read_form_value(self, prompt: str, default: str = "") -> str:
        if not self.interactive:
            raise UsageError("Setup requires an interactive terminal")
        suffix = f" [{default}]" if default else ""
        value = self.read_line(f"{prompt}{suffix}: ")
        if value is None:
            raise UsageError("Setup input ended before confirmation")
        return value.strip() or default

    def choose(self, prompt: str, options: list[tuple[str, str]], *, allow_back: bool = False, allow_skip: bool = False) -> str:
        if not self.interactive:
            raise UsageError("Setup requires an interactive terminal")
        self.stdout.write(prompt + "\n")
        for index, (_, label) in enumerate(options, 1):
            self.stdout.write(f"  {index}. {label}\n")
        controls = ["cancel"]
        if allow_back:
            controls.append("back")
        if allow_skip:
            controls.append("skip")
        self.stdout.write(f"  ({'/'.join(controls)})\n")
        self.stdout.flush()
        while True:
            value = self.read_form_value("Select").lower()
            if value in controls:
                return value
            if value.isdigit() and 1 <= int(value) <= len(options):
                return options[int(value) - 1][0]
            self.stderr.write("Enter a listed number or control word.\n")
            self.stderr.flush()


class InteractiveTerminal(PlainTerminal):
    def __init__(self, history_file, completer, *, no_color: bool = False) -> None:
        super().__init__(interactive=True, color=not no_color)
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import Completer, Completion
            from prompt_toolkit.history import DummyHistory, FileHistory

            class SlashCompleter(Completer):
                def get_completions(self, document, complete_event):
                    text = document.text_before_cursor
                    if text.startswith("/") and " " not in text:
                        for value in completer(text):
                            yield Completion(value, start_position=-len(text))

            self._session = PromptSession(history=FileHistory(str(history_file)), completer=SlashCompleter(), complete_while_typing=False)
            # prompt_toolkit does not provide a per-prompt ``add_history``
            # switch. Keep setup fields and secrets on a separate session whose
            # history intentionally discards every value.
            self._form_session = PromptSession(history=DummyHistory())
            self._secret_session = PromptSession(history=DummyHistory())
        except ImportError:
            self._session = None
            self._form_session = None
            self._secret_session = None

    def read_line(self, prompt: str = "AutoMemory> ") -> str | None:
        if self._session:
            try:
                return self._session.prompt(prompt)
            except EOFError:
                return None
        return super().read_line(prompt)

    def read_secret(self, prompt: str) -> str:
        if self._secret_session:
            return self._secret_session.prompt(prompt, is_password=True)
        return super().read_secret(prompt)

    def read_form_value(self, prompt: str, default: str = "") -> str:
        if self._form_session:
            suffix = f" [{default}]" if default else ""
            try:
                value = self._form_session.prompt(f"{prompt}{suffix}: ", is_password=False)
            except EOFError as exc:
                raise UsageError("Setup input ended before confirmation") from exc
            return value.strip() or default
        return super().read_form_value(prompt, default)
