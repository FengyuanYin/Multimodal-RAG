"""Shared command parsing helpers."""

from __future__ import annotations

from ..errors import UsageError


def pop_flag(args: list[str], name: str) -> bool:
    if name in args:
        args.remove(name)
        return True
    return False


def pop_option(args: list[str], name: str, default: str | None = None) -> str | None:
    if name not in args:
        return default
    index = args.index(name)
    if index + 1 >= len(args):
        raise UsageError(f"{name} requires a value")
    value = args[index + 1]
    del args[index:index + 2]
    return value


def require_count(args: list[str], minimum: int, usage: str) -> None:
    if len(args) < minimum:
        raise UsageError(f"Usage: {usage}")


def resolve_prefix(items: list[dict], value: str, label: str) -> str:
    exact = [item["id"] for item in items if item["id"] == value]
    if exact:
        return exact[0]
    matches = [item["id"] for item in items if item["id"].startswith(value)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise UsageError(f"{label} not found: {value}")
    raise UsageError(f"{label} prefix is ambiguous: {value}")
