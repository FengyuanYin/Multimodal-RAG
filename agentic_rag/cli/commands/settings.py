"""Safe non-secret configuration and credential commands."""

from __future__ import annotations

import json
from typing import Any

from ..config import AutoMemoryConfig
from ..credentials import ENV_NAMES
from ..errors import UsageError
from ..models import CommandResult, CommandSpec, EventKind, OutputEvent
from .utils import require_count


def _lookup(value: dict, path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise UsageError(f"Unknown config key: {path}")
        current = current[part]
    return current


def _assign(value: dict, path: str, item: Any) -> None:
    parts, current = path.split("."), value
    for part in parts[:-1]:
        if not isinstance(current.get(part), dict):
            raise UsageError(f"Unknown config key: {path}")
        current = current[part]
    if parts[-1] not in current:
        raise UsageError(f"Unknown config key: {path}")
    current[parts[-1]] = item


def config(ctx, args, output, cancel, router):
    action = args.pop(0).lower() if args else "list"
    payload = ctx.config.to_dict()
    if action == "list":
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    elif action == "get":
        require_count(args, 1, "/config get <key>")
        text = json.dumps(_lookup(payload, args[0]), ensure_ascii=False)
    elif action == "set":
        require_count(args, 2, "/config set <key> <value>")
        raw = " ".join(args[1:])
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            item = raw
        _assign(payload, args[0], item)
        candidate = AutoMemoryConfig.from_dict(payload)
        ctx.save_config(candidate)
        text = f"Saved config: {args[0]}"
    elif action == "unset":
        require_count(args, 1, "/config unset <key>")
        default = AutoMemoryConfig().to_dict()
        _assign(payload, args[0], _lookup(default, args[0]))
        ctx.save_config(AutoMemoryConfig.from_dict(payload))
        text = f"Reset config: {args[0]}"
    elif action == "test":
        service = args[0] if args else "all"
        status = {item.name.lower(): item.status for item in ctx.diagnostics.report()}
        selected = status if service == "all" else {service: status.get(service, "unknown")}
        text = json.dumps(selected, ensure_ascii=False, indent=2)
    else:
        raise UsageError("Usage: /config [list|get|set|unset|test] ...")
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text)


def secret(ctx, args, output, cancel, router):
    action = args.pop(0).lower() if args else "status"
    if action == "status":
        text = "\n".join(f"{name}: {ctx.credentials.source(name)}" for name in ENV_NAMES)
    elif action == "set":
        require_count(args, 1, "/secret set <name>")
        if len(args) != 1:
            raise UsageError("Do not place credentials in command arguments", hint="Use /secret set <name> and enter the value at the hidden prompt")
        name = args[0]
        if name not in ENV_NAMES:
            raise UsageError(f"Unknown credential name: {name}")
        value = output.read_secret(f"{name}: ")
        ctx.credentials.set(name, value, persist=True)
        ctx.reload_services()
        text = f"Credential saved for {name}; source={ctx.credentials.source(name)}"
    elif action == "delete":
        require_count(args, 1, "/secret delete <name>")
        if args[0] not in ENV_NAMES:
            raise UsageError(f"Unknown credential name: {args[0]}")
        deleted = ctx.credentials.delete(args[0])
        ctx.reload_services()
        text = f"Credential {'deleted' if deleted else 'was not stored'}: {args[0]}"
    elif action == "test":
        require_count(args, 1, "/secret test <name>")
        if args[0] not in ENV_NAMES:
            raise UsageError(f"Unknown credential name: {args[0]}")
        text = f"{args[0]}: {ctx.credentials.source(args[0])}"
    else:
        raise UsageError("Usage: /secret [status|set|delete|test] ...")
    output.emit(OutputEvent(EventKind.RESULT, text=text))
    return CommandResult(text=text)


def register(router) -> None:
    router.register(CommandSpec("config", "View or change non-secret configuration", "/config [list|get|set|unset|test] ...", config, group="Settings"))
    router.register(CommandSpec("secret", "Manage cloud credentials securely", "/secret set <name> | status | delete <name> | test <name>", secret, group="Settings"))
