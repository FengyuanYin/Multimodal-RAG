"""Shared application context for AutoMemory workspaces."""

from __future__ import annotations

from dataclasses import dataclass

from .config import SecretStore
from .models import AutoMemoryConfig
from .paths import AutoMemoryPaths


@dataclass
class AutoMemoryContext:
    paths: AutoMemoryPaths
    config: AutoMemoryConfig
    secrets: SecretStore
    state: object
    runtime: object
    chat: object
    knowledge: object
    evaluation: object
    web: object
    mineru: object
    diagnostics: object
