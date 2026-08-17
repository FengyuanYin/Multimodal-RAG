"""Dependency-free AutoMemory startup branding."""

from __future__ import annotations

import re


FULL_LOGO = (
    "    _         _        __  __                              ",
    "   / \\  _   _| |_ ___ |  \\/  | ___ _ __ ___   ___  _ __ _   _",
    "  / _ \\| | | | __/ _ \\| |\\/| |/ _ \\ '_ ` _ \\ / _ \\| '__| | | |",
    " / ___ \\ |_| | || (_) | |  | |  __/ | | | | | (_) | |  | |_| |",
    "/_/   \\_\\__,_|\\__\\___/|_|  |_|\\___|_| |_| |_|\\___/|_|   \\__, |",
    "                                                         |___/ ",
)
COMPACT_LOGO = "AutoMemory"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")


def _gradient_line(line: str) -> str:
    visible = max(1, len(line.rstrip()))
    output: list[str] = []
    for index, char in enumerate(line.rstrip()):
        ratio = index / max(1, visible - 1)
        red = round(124 + (25 - 124) * ratio)
        green = round(58 + (211 - 58) * ratio)
        blue = round(237 + (238 - 237) * ratio)
        output.append(f"\x1b[38;2;{red};{green};{blue}m{char}")
    return "".join(output) + "\x1b[0m"


def sanitize_summary(value: str) -> str:
    value = CONTROL_RE.sub(" ", str(value)).strip()
    value = value.split("?", 1)[0]
    value = re.sub(r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*\S+", r"\1=[REDACTED]", value)
    return value[:100]


def render_startup_banner(*, width: int, color: bool, version: str, llm_summary: str, needs_setup: bool, knowledge_base: str = "default", rag_mode: str = "balanced") -> str:
    logo = FULL_LOGO if width >= max(len(line) for line in FULL_LOGO) else (COMPACT_LOGO,)
    rendered = [_gradient_line(line) for line in logo] if color else [line.rstrip() for line in logo]
    rendered.append(f"AutoMemory {sanitize_summary(version)} | {sanitize_summary(llm_summary)}")
    rendered.append(f"Knowledge base: {sanitize_summary(knowledge_base)} | RAG mode: {sanitize_summary(rag_mode)}")
    if needs_setup:
        rendered.append("LLM is not configured. Run /setup to connect a cloud API.")
    rendered.append("Type normally to chat | /s <question> for knowledge | /help for commands")
    return "\n".join(rendered) + "\n\n"


def visible_width(value: str) -> int:
    return max((len(ANSI_RE.sub("", line)) for line in value.splitlines()), default=0)
