"""Direct and knowledge-grounded cloud streaming chat."""

from __future__ import annotations

import re
import time

from ..cancellation import CancellationToken
from ..errors import CancelledError, ConfigurationError
from ..models import EventKind, OutputEvent


class DirectChatService:
    """Direct chat deliberately has no retriever dependency."""

    def __init__(self, state, llm_client, config) -> None:
        self.state, self.llm_client, self.config = state, llm_client, config

    def stream(self, conversation_id: str, question: str, output, cancel: CancellationToken) -> dict:
        return self._stream(conversation_id, question, output, cancel, mode="direct", system="You are AutoMemory, a helpful and concise AI assistant.", user_content=question, sources=[])

    def _stream(self, conversation_id: str, question: str, output, cancel: CancellationToken, *, mode: str, system: str, user_content: str, sources: list[dict], extra_metadata: dict | None = None) -> dict:
        question = question.strip()
        if not question:
            raise ConfigurationError("Question is required")
        if not self.llm_client:
            raise ConfigurationError("Cloud LLM is not configured", hint="Run /secret set llm_api_key and /config test llm")
        history = self.state.list_messages(conversation_id, 24)
        self.state.append_message(conversation_id, "user", question, mode, "complete")
        assistant = self.state.append_message(conversation_id, "assistant", "", mode, "streaming")
        memories = self.state.list_memories(enabled_only=True) if self.config.memory_enabled else []
        if memories:
            system += "\n\nEnabled user memories:\n" + "\n".join(f"- {item['content']}" for item in memories)
        messages = [{"role": "system", "content": system}]
        messages.extend({"role": item["role"], "content": item["content"]} for item in history if item["role"] in {"user", "assistant"} and item["content"])
        messages.append({"role": "user", "content": user_content})
        parts, started = [], time.perf_counter()
        try:
            for delta in self.llm_client.stream_chat(messages, cancel):
                cancel.checkpoint()
                parts.append(delta)
                output.emit(OutputEvent(EventKind.DELTA, text=delta))
            output.emit(OutputEvent(EventKind.RESULT))
            answer = "".join(parts)
            if sources:
                answer = self._validate_citations(answer, len(sources))
            metadata = {"mode": mode, "sources": sources, "latency_ms": round((time.perf_counter() - started) * 1000, 2), **(extra_metadata or {})}
            self.state.finalize_message(assistant["id"], "complete", answer, metadata)
            if sources:
                output.emit(OutputEvent(EventKind.SOURCES, data=sources))
            return {"answer": answer, "sources": sources, "metadata": metadata}
        except CancelledError:
            answer = "".join(parts)
            self.state.finalize_message(assistant["id"], "interrupted", answer, {"mode": mode, "sources": sources})
            raise
        except Exception:
            answer = "".join(parts)
            self.state.finalize_message(assistant["id"], "error", answer, {"mode": mode, "sources": sources})
            raise

    @staticmethod
    def _validate_citations(answer: str, source_count: int) -> str:
        def replace(match: re.Match[str]) -> str:
            number = int(match.group(1))
            return match.group(0) if 1 <= number <= source_count else ""
        return re.sub(r"\[(\d+)\]", replace, answer)


class GroundedChatService:
    def __init__(self, state, llm_client, config, retriever) -> None:
        self.state, self.llm_client, self.config, self.retriever = state, llm_client, config, retriever
        self._direct_core = DirectChatService(state, llm_client, config)

    def stream(self, conversation_id: str, question: str, output, cancel: CancellationToken, scope: str = "all") -> dict:
        result = self.retriever.search(question, scope, self.config.retrieval_mode, self.config.top_k, cancel)
        if not result.hits:
            answer = "No matching evidence was found in the selected knowledge scope."
            self.state.append_message(conversation_id, "user", question, "rag", "complete")
            self.state.append_message(conversation_id, "assistant", answer, "rag", "complete", {"sources": [], "retrieval_trace": result.trace})
            output.emit(OutputEvent(EventKind.RESULT, text=answer))
            return {"answer": answer, "sources": [], "metadata": {"retrieval_trace": result.trace, "no_match": True}}
        sources = []
        blocks = []
        for index, hit in enumerate(result.hits, 1):
            source = {"index": index, "target_id": hit.target_id, "document_id": hit.document_id, "document": hit.document, "page": hit.page, "score": hit.score, "modality": hit.modality, "media_refs": hit.media_refs, "text": hit.text}
            sources.append(source)
            blocks.append(f"[{index}] Document: {hit.document}; page: {hit.page}; modality: {hit.modality}\n{hit.text}")
        system = "You are AutoMemory, a rigorous document assistant. Answer only from supplied evidence. Cite every factual claim with [n]. If evidence is insufficient, say so. Never invent a citation."
        user_content = "Evidence:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
        return self._direct_core._stream(
            conversation_id,
            question,
            output,
            cancel,
            mode="rag",
            system=system,
            user_content=user_content,
            sources=sources,
            extra_metadata={"retrieval_trace": result.trace},
        )
