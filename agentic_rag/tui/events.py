"""Worker cancellation and typed progress events."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Any, Callable


class JobCancelled(RuntimeError):
    pass


class CancelToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def checkpoint(self) -> None:
        if self.cancelled:
            raise JobCancelled("operation cancelled")


@dataclass(frozen=True)
class JobProgress:
    job_id: str
    kind: str
    phase: str
    message: str
    completed: int | None = None
    total: int | None = None


@dataclass(frozen=True)
class StreamDelta:
    job_id: str
    conversation_id: str
    message_id: str
    text: str


@dataclass(frozen=True)
class JobResultEvent:
    job_id: str
    status: str
    value: Any = None
    safe_error: str = ""


EventCallback = Callable[[JobProgress | StreamDelta | JobResultEvent], None]
