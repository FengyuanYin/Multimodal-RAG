"""Slash-command registration."""

from . import core, evaluation, knowledge, memory, mineru, sessions, settings, web


def register_all(router) -> None:
    for module in (core, sessions, memory, knowledge, web, mineru, evaluation, settings):
        module.register(router)


__all__ = ["register_all"]
