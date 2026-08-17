"""LLM-authored compaction of mutable workspace history."""

from __future__ import annotations

from ..errors import ConfigurationError
from ..models import EventKind, OutputEvent


SUMMARY_START = '<AUTOMEMORY_HISTORY_SUMMARY version="1" original_history="false">'
SUMMARY_END = "</AUTOMEMORY_HISTORY_SUMMARY>"


class WorkspaceCompactionService:
    def __init__(self, llm_client, files, repository, config) -> None:
        self.llm_client, self.files, self.repository, self.config = llm_client, files, repository, config

    def compact(self, workspace: dict, events: list[dict], output, cancel) -> list[dict]:
        keep_count = self.config.document_recent_turns * 2
        if len(events) <= keep_count:
            raise ConfigurationError("The full document and recent history exceed the input budget", hint="Leave the workspace or use a larger-context main LLM")
        old, recent = events[:-keep_count], events[-keep_count:]
        output.emit(OutputEvent(EventKind.PROGRESS, phase="compaction", text="Summarizing older document-workspace history"))
        payload = "\n\n".join(f'{item["role"].upper()}: {item["content"]}' for item in old)
        messages = [{"role":"system","content":"Summarize only the supplied conversation history. Preserve facts, user intent, unfinished tasks, conclusions, image IDs and file IDs. This is not the source document."},{"role":"user","content":payload}]
        summary = "".join(self.llm_client.stream_chat(messages, cancel, temperature=0.0)).strip()
        if not summary: raise ConfigurationError("History compaction returned an empty summary")
        tagged = f"{SUMMARY_START}\n{summary}\n{SUMMARY_END}"
        record = self.files.create_markdown(workspace["id"], "summary", "history-summary.md", tagged, "Automatic history compaction")
        self.repository.append_event(workspace["id"], "system", tagged, "summary", metadata={"covers_through_sequence": old[-1]["sequence"], "file_id": record["id"]}, file_id=record["id"])
        return [{"role":"system", "content":tagged, "sequence":old[-1]["sequence"], "event_kind":"summary", "status":"complete"}, *recent]
