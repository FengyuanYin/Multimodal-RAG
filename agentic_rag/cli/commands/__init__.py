"""Slash-command registration."""

from . import core, document_context, evaluation, graph, knowledge, knowledge_bases, memory, mineru, modes, sessions, settings, web


def register_all(router) -> None:
    for module in (core, modes, knowledge_bases, document_context, graph, sessions, memory, knowledge, web, mineru, evaluation, settings):
        module.register(router)


__all__ = ["register_all"]
