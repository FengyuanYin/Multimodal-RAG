"""Direct and grounded streaming chat for AutoMemory."""

from __future__ import annotations

import time
import uuid

from ..events import CancelToken, EventCallback, JobCancelled, StreamDelta
from ..models import ChatRequest, ChatResult, MessageStatus, RetrievalMode


class ChatService:
    def __init__(self, runtime, state) -> None:
        self.runtime = runtime
        self.state = state

    def stream(self, request: ChatRequest, emit: EventCallback | None = None, cancel: CancelToken | None = None, job_id: str | None = None) -> ChatResult:
        cancel = cancel or CancelToken()
        job_id = job_id or f"chat_{uuid.uuid4().hex}"
        question = request.question.strip()
        if not question:
            raise ValueError("question is required")
        client = self.runtime.orchestrator.llm_client
        if client is None:
            raise RuntimeError("LLM API key is not configured")
        self.state.append_message(request.conversation_id, "user", question, request.mode, MessageStatus.COMPLETE.value)
        assistant = self.state.append_message(request.conversation_id, "assistant", "", request.mode, MessageStatus.STREAMING.value)
        history = self.state.list_messages(request.conversation_id, limit=24)[:-2]
        memory_text = "\n".join(f"- {item.content}" for item in self.state.list_memories(enabled_only=True)) if self.runtime.config.memory_enabled else ""
        sources, metadata = [], {"mode": request.mode}
        if request.mode == "rag":
            sources, trace = self._retrieve(question, request.collection_id)
            if not sources:
                answer = "No matching content was found in the selected knowledge scope."
                self.state.update_message(assistant.id, MessageStatus.COMPLETE.value, answer, {"mode": "rag", "sources": []})
                return ChatResult(answer, [], {"mode": "rag", "no_match": True})
            context = "\n\n".join(f"[{index}] ({item.get('document','document')} p.{item.get('page',1)})\n{item['content']}" for index, item in enumerate(sources, 1))
            system = "You are AutoMemory, a rigorous document assistant. Answer only from the supplied sources. If evidence is insufficient, say so. Cite sources as [1], [2]."
            user_content = f"Sources:\n{context}\n\nQuestion: {question}"
            metadata.update({"sources": sources, "retrieval_trace": trace})
        else:
            system = "You are AutoMemory, a helpful and concise AI assistant. Answer the user directly."
            user_content = question
        if memory_text:
            system += f"\n\nRelevant user memories:\n{memory_text}"
        messages = [{"role": "system", "content": system}]
        messages.extend({"role": item.role, "content": item.content} for item in history if item.role in {"user", "assistant"} and item.content)
        messages.append({"role": "user", "content": user_content})
        started, parts = time.perf_counter(), []
        try:
            stream = client.chat.completions.create(model=self.runtime.config.llm_model, messages=messages, temperature=0.2, stream=True)
            for chunk in stream:
                cancel.checkpoint()
                choices = getattr(chunk, "choices", None) or []
                delta = getattr(getattr(choices[0], "delta", None), "content", "") if choices else ""
                if delta:
                    parts.append(delta)
                    if emit:
                        emit(StreamDelta(job_id, request.conversation_id, assistant.id, delta))
            answer = "".join(parts)
            metadata["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self.state.update_message(assistant.id, MessageStatus.COMPLETE.value, answer, metadata)
            return ChatResult(answer, sources, metadata)
        except JobCancelled:
            answer = "".join(parts)
            self.state.update_message(assistant.id, MessageStatus.INTERRUPTED.value, answer, metadata)
            raise
        except Exception:
            answer = "".join(parts)
            self.state.update_message(assistant.id, MessageStatus.ERROR.value, answer, metadata)
            raise

    def _retrieve(self, query: str, collection_id: str) -> tuple[list[dict], dict]:
        mode = self.runtime.config.retrieval_mode
        use_vector = mode in {RetrievalMode.VECTOR.value, RetrievalMode.HYBRID.value, RetrievalMode.MULTIMODAL.value}
        use_keyword = mode in {RetrievalMode.KEYWORD.value, RetrievalMode.HYBRID.value, RetrievalMode.MULTIMODAL.value}
        candidates = self.runtime.retriever.retrieve(query, top_k=min(50, self.runtime.config.top_k * 5), use_vector=use_vector, use_keyword=use_keyword)
        if collection_id and collection_id != "all":
            candidates = [item for item in candidates if item.metadata.get("category_id") == collection_id]
        reranker = getattr(self.runtime.orchestrator, "reranker", None)
        if reranker and candidates:
            candidates = reranker.rerank(query, candidates, top_k=self.runtime.config.top_k)
        candidates = candidates[: self.runtime.config.top_k]
        documents = {item["id"]: item for item in self.runtime.repository.list_documents(include_unsearchable=True)}
        sources = []
        for item in candidates:
            document_id = item.metadata.get("document_id") or item.metadata.get("doc_id") or item.doc_id.split("_chunk_")[0]
            document = documents.get(document_id, {})
            sources.append({
                "doc_id": item.doc_id, "document_id": document_id,
                "document": document.get("name", item.metadata.get("source", "document")),
                "content": item.content, "score": float(item.score),
                "page": int(item.metadata.get("page", 1) or 1), "modality": item.modality,
                "media_refs": getattr(item, "media_refs", []) or item.metadata.get("media_refs", []),
            })
        trace = self.runtime.retriever.last_trace.to_dict() if self.runtime.retriever.last_trace else {}
        if mode == RetrievalMode.MULTIMODAL.value and candidates:
            metadata_media = self.runtime.retriever.retrieve_media(candidates, include_data=False)
            trace["media"] = metadata_media
        return sources, trace
