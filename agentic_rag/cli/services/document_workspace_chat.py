"""Full immutable-Markdown chat with bounded file and image tools."""

from __future__ import annotations

import json
import time

from ..errors import CancelledError, ConfigurationError, UsageError
from ..models import EventKind, OutputEvent


TOOLS = [
    {"type":"function","function":{"name":"read_image","description":"Analyze one image attached to the active Markdown document.","parameters":{"type":"object","properties":{"media_id":{"type":"string"},"purpose":{"type":"string"}},"required":["media_id","purpose"],"additionalProperties":False}}},
    {"type":"function","function":{"name":"read_file","description":"Read a bounded range from a generated workspace Markdown file.","parameters":{"type":"object","properties":{"file_id":{"type":"string"},"start":{"type":"integer"},"max_chars":{"type":"integer"}},"required":["file_id"],"additionalProperties":False}}},
    {"type":"function","function":{"name":"write_markdown","description":"Create a new Markdown note in the active workspace.","parameters":{"type":"object","properties":{"name":{"type":"string"},"content":{"type":"string"},"purpose":{"type":"string"}},"required":["name","content","purpose"],"additionalProperties":False}}},
]


class DocumentWorkspaceChatService:
    def __init__(self, state, knowledge, artifacts, llm_client, builder, budget, compaction, files, images, long_responses, config) -> None:
        self.state, self.repo, self.knowledge, self.artifacts = state, state.workspaces, knowledge, artifacts
        self.llm_client, self.builder, self.budget, self.compaction = llm_client, builder, budget, compaction
        self.files, self.images, self.long_responses, self.config = files, images, long_responses, config

    def stream(self, workspace_id: str, question: str, output, cancel) -> dict:
        question = question.strip()
        if not question: raise ConfigurationError("Document question is required")
        if not self.llm_client: raise ConfigurationError("Cloud LLM is not configured")
        workspace = self.repo.get(workspace_id)
        if not workspace or workspace["status"] != "ready": raise ConfigurationError("Document workspace is unavailable")
        document = self.knowledge.get_document(workspace["document_id"])
        artifact = self.knowledge.get_document_artifact(workspace["document_id"])
        if not document or not artifact or artifact["checksum"] != workspace["markdown_checksum"]: raise ConfigurationError("Document workspace source has changed or was deleted")
        markdown = self.artifacts.verify(artifact).read_text("utf-8")
        fixed = self.builder.build_fixed_prefix(document, artifact, markdown, document["media"])
        self.budget.assert_fixed_fits(fixed, question)
        events = self.repo.context_events(workspace_id)
        initial_budget = self.budget.calculate(fixed, "", events, question)
        if initial_budget.requires_compaction:
            self.compaction.compact(workspace, events, output, cancel)
            events = self.repo.context_events(workspace_id)
        final_budget = self.budget.calculate(fixed, "", events, question)
        used = final_budget.fixed_prefix_tokens + final_budget.summary_tokens + final_budget.recent_history_tokens + final_budget.tool_tokens + final_budget.question_tokens
        if used > final_budget.hard_input_limit: raise ConfigurationError("Document workspace still exceeds the input budget after compaction")
        self.repo.append_event(workspace_id, "user", question)
        assistant = self.repo.append_event(workspace_id, "assistant", "", status="streaming")
        messages = fixed + self.builder.build_variable_suffix(events, question)
        parts, usage, tool_trace, started = [], {}, [], time.perf_counter()
        vlm_calls = writes = 0
        try:
            for round_index in range(self.config.document_tool_round_limit):
                calls = []
                for event in self.llm_client.stream_chat_events(messages, TOOLS, cancel, max_tokens=self.config.document_output_reserve_tokens):
                    if event.kind == "text_delta": parts.append(event.text); output.emit(OutputEvent(EventKind.DELTA, text=event.text))
                    elif event.kind == "tool_call": calls.append(event)
                    elif event.kind == "usage": usage.update(event.usage)
                if not calls: break
                assistant_tool_calls = []
                for call in calls:
                    assistant_tool_calls.append({"id":call.tool_call_id,"type":"function","function":{"name":call.tool_name,"arguments":json.dumps(call.arguments,ensure_ascii=False)}})
                messages.append({"role":"assistant","content":"","tool_calls":assistant_tool_calls})
                for call in calls:
                    self.repo.append_event(workspace_id, "assistant", json.dumps({"name":call.tool_name,"arguments":call.arguments},ensure_ascii=False), "tool_call")
                    try:
                        if call.tool_name == "read_image":
                            vlm_calls += 1
                            if vlm_calls > self.config.document_vlm_call_limit: raise UsageError("Per-turn VLM call limit reached")
                            result = self.images.analyze(workspace, str(call.arguments.get("media_id", "")), str(call.arguments.get("purpose", "")), output, cancel)
                        elif call.tool_name == "read_file":
                            result = json.dumps(self.files.read_text(workspace_id, str(call.arguments.get("file_id", "")), int(call.arguments.get("start",0)), int(call.arguments.get("max_chars",48000))),ensure_ascii=False)
                        elif call.tool_name == "write_markdown":
                            writes += 1
                            if writes > self.config.document_write_call_limit: raise UsageError("Per-turn file write limit reached")
                            record = self.files.create_markdown(workspace_id,"model_note",str(call.arguments.get("name","note.md")),str(call.arguments.get("content","")),str(call.arguments.get("purpose","")))
                            result = json.dumps({"file_id":record["id"],"name":record["display_name"]},ensure_ascii=False)
                        else: raise UsageError(f"Unknown workspace tool: {call.tool_name}")
                    except Exception as exc:
                        result = json.dumps({"error":type(exc).__name__,"message":str(exc)},ensure_ascii=False)
                    self.repo.append_event(workspace_id, "tool", result, "tool_result", metadata={"tool_call_id":call.tool_call_id,"tool_name":call.tool_name})
                    messages.append({"role":"tool","tool_call_id":call.tool_call_id,"content":result})
                    tool_trace.append(call.tool_name)
            else:
                raise ConfigurationError("Document tool loop limit reached")
            output.emit(OutputEvent(EventKind.RESULT))
            answer = "".join(parts)
            stored, file_id, answer_meta = self.long_responses.finalize(workspace_id, answer)
            metadata = {"mode":"document","budget":final_budget.__dict__ if hasattr(final_budget,"__dict__") else {name:getattr(final_budget,name) for name in final_budget.__dataclass_fields__},"usage":usage,"tools":tool_trace,"prefix_fingerprint":self.builder.prefix_fingerprint(fixed),"latency_ms":round((time.perf_counter()-started)*1000,2),**answer_meta}
            self.repo.finalize_event(assistant["id"], "complete", stored, metadata, file_id)
            return {"answer":answer,"metadata":metadata}
        except CancelledError:
            self.repo.finalize_event(assistant["id"], "interrupted", "".join(parts), {"mode":"document"}); raise
        except Exception:
            self.repo.finalize_event(assistant["id"], "error", "".join(parts), {"mode":"document"}); raise
