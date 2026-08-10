"""Cooperative cancellation for foreground CLI work."""

from __future__ import annotations

from threading import Event

from .errors import CancelledError


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def checkpoint(self) -> None:
        if self.cancelled:
            raise CancelledError()
