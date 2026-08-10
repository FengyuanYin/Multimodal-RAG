"""Deterministic slash-command and direct/RAG input routing."""

from __future__ import annotations

from difflib import get_close_matches

from .errors import UsageError
from .models import CommandSpec, InputKind, ParsedInput


def tokenize_windows(text: str) -> list[str]:
    """Split command arguments while preserving Windows path backslashes."""
    tokens, current, quoted = [], [], False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            quoted = not quoted
        elif char.isspace() and not quoted:
            if current:
                tokens.append("".join(current))
                current = []
        elif char == "\\" and quoted and index + 1 < len(text) and text[index + 1] == '"':
            current.append('"')
            index += 1
        else:
            current.append(char)
        index += 1
    if quoted:
        raise UsageError("Unclosed double quote in command")
    if current:
        tokens.append("".join(current))
    return tokens


class CommandRouter:
    def __init__(self) -> None:
        self._commands: dict[str, CommandSpec] = {}
        self._canonical: dict[str, CommandSpec] = {}

    def register(self, spec: CommandSpec) -> None:
        names = (spec.name, *spec.aliases)
        for name in names:
            key = name.lower().lstrip("/")
            if key in self._commands:
                raise ValueError(f"Duplicate command: {key}")
            self._commands[key] = spec
        self._canonical[spec.name] = spec

    def parse(self, text: str) -> ParsedInput:
        raw, stripped = text, text.lstrip()
        if not stripped:
            return ParsedInput(InputKind.EMPTY, raw)
        if raw == "/s" or raw.startswith("/s ") or raw.startswith("/s\t"):
            return ParsedInput(InputKind.RAG_CHAT, raw, name="s", question=raw[2:].strip())
        if raw.startswith("/"):
            tokens = tokenize_windows(raw)
            if not tokens:
                return ParsedInput(InputKind.EMPTY, raw)
            return ParsedInput(InputKind.COMMAND, raw, name=tokens[0][1:].lower(), arguments=tokens[1:])
        return ParsedInput(InputKind.DIRECT_CHAT, raw, question=stripped)

    def resolve(self, name: str) -> CommandSpec:
        key = name.lower().lstrip("/")
        spec = self._commands.get(key)
        if spec:
            return spec
        suggestion = self.suggest(key)
        hint = f" Did you mean /{suggestion}?" if suggestion else " Run /help for available commands."
        raise UsageError(f"Unknown command: /{key}.{hint}")

    def suggest(self, name: str) -> str:
        matches = get_close_matches(name, sorted(self._commands), n=1, cutoff=0.55)
        return matches[0] if matches else ""

    def complete(self, prefix: str) -> list[str]:
        prefix = prefix.lower().lstrip("/")
        return [f"/{name}" for name in sorted(self._commands) if name.startswith(prefix)]

    def specs(self) -> list[CommandSpec]:
        return sorted(self._canonical.values(), key=lambda item: (item.group, item.name))
