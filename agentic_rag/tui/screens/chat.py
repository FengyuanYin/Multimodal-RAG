"""Streaming direct and knowledge-grounded chat workspace."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, ListItem, ListView, Markdown, Select, Static

from ..events import CancelToken, JobCancelled, StreamDelta
from ..models import ChatRequest
from ..widgets import Transcript, WorkspaceNav
from .base import WorkspaceScreen


class ChatScreen(WorkspaceScreen):
    BINDINGS = [("ctrl+enter", "send", "Send"), ("escape", "stop", "Stop")]

    def __init__(self) -> None:
        super().__init__()
        self.conversation_id = ""
        self.cancel_token: CancelToken | None = None
        self._assistant = None
        self._answer = ""
        self.selected_memory_id = ""

    def compose(self) -> ComposeResult:
        yield WorkspaceNav()
        with Vertical(classes="workspace"):
            yield Label("Chat", classes="title")
            with Horizontal(classes="toolbar"):
                yield Select([("Direct chat", "direct"), ("Knowledge RAG", "rag")], value="direct", id="chat-mode", allow_blank=False)
                yield Select([("All collections", "all")], value="all", id="chat-collection", allow_blank=False)
                yield Button("New", id="new-conversation")
                yield Button("Stop", id="stop-chat", variant="warning", disabled=True)
            with Horizontal(classes="body-row"):
                with Vertical(id="chat-sidebar"):
                    yield Label("Conversations", classes="section-title")
                    yield ListView(id="conversations")
                    yield Input(placeholder="Rename conversation", id="conversation-title")
                    with Horizontal(classes="compact-actions"):
                        yield Button("Rename", id="rename-conversation")
                        yield Button("Clear", id="clear-conversation")
                        yield Button("Delete", id="delete-conversation", variant="error")
                    yield Label("Long-term memory", classes="section-title")
                    yield ListView(id="memories")
                    yield Input(placeholder="Add a memory", id="memory-input")
                    with Horizontal(classes="compact-actions"):
                        yield Button("Add", id="add-memory")
                        yield Button("Enable/disable", id="toggle-memory")
                        yield Button("Delete", id="delete-memory", variant="error")
                with Vertical(id="chat-main"):
                    yield Transcript(id="transcript")
                    yield Markdown("", id="chat-sources")
                    yield Static("Direct mode · ready", id="chat-status", classes="status-line")
                    with Horizontal(classes="composer"):
                        yield Input(placeholder="Ask anything…", id="chat-input")
                        yield Button("Send", id="send-chat", variant="primary")

    def on_mount(self) -> None:
        self.refresh_conversations()
        self.refresh_memories()
        self.refresh_collections()

    def refresh_conversations(self) -> None:
        view = self.query_one("#conversations", ListView)
        view.clear()
        conversations = self.app.ctx.state.list_conversations()
        if not conversations:
            conversations = [self.app.ctx.state.create_conversation()]
        for item in conversations:
            row = ListItem(Label(item.title), id=f"conversation-{item.id}")
            row.data = item.id
            view.append(row)
        self.load_conversation(conversations[0].id)

    def refresh_collections(self) -> None:
        options = [("All collections", "all")]
        options.extend((item.name, item.id) for item in self.app.ctx.state.list_categories())
        select = self.query_one("#chat-collection", Select)
        select.set_options(options)
        select.value = "all"

    def refresh_memories(self) -> None:
        view = self.query_one("#memories", ListView)
        view.clear()
        for item in self.app.ctx.state.list_memories():
            prefix = "●" if item.enabled else "○"
            row = ListItem(Label(f"{prefix} {item.content}"), id=f"memory-{item.id}")
            row.data = item.id
            view.append(row)

    def load_conversation(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        transcript = self.query_one(Transcript)
        transcript.remove_children()
        messages = self.app.ctx.state.list_messages(conversation_id)
        if not messages:
            transcript.empty_state("Start a direct conversation, or switch to Knowledge RAG for cited answers.")
        else:
            for item in messages:
                transcript.add_message(item.role, item.content, item.status if item.status != "complete" else "")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id == "conversations":
            self.load_conversation(event.item.data)
        elif event.list_view.id == "memories":
            self.selected_memory_id = event.item.data

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self.action_send()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-chat":
            self.action_send()
        elif event.button.id == "stop-chat":
            self.action_stop()
        elif event.button.id == "new-conversation":
            conversation = self.app.ctx.state.create_conversation()
            self.refresh_conversations()
            self.load_conversation(conversation.id)
        elif event.button.id == "clear-conversation" and self.conversation_id:
            self.app.ctx.state.clear_conversation(self.conversation_id)
            self.load_conversation(self.conversation_id)
        elif event.button.id == "rename-conversation" and self.conversation_id:
            try:
                self.app.ctx.state.rename_conversation(self.conversation_id, self.query_one("#conversation-title", Input).value)
                self.query_one("#conversation-title", Input).value = ""
                self.refresh_conversations()
            except Exception as exc:
                self.notify_error(exc)
        elif event.button.id == "delete-conversation" and self.conversation_id:
            self.app.ctx.state.delete_conversation(self.conversation_id)
            self.refresh_conversations()
        elif event.button.id == "add-memory":
            field = self.query_one("#memory-input", Input)
            try:
                self.app.ctx.state.create_memory(field.value)
                field.value = ""
                self.refresh_memories()
            except Exception as exc:
                self.notify_error(exc)
        elif event.button.id == "delete-memory" and self.selected_memory_id:
            self.app.ctx.state.delete_memory(self.selected_memory_id)
            self.selected_memory_id = ""
            self.refresh_memories()
        elif event.button.id == "toggle-memory" and self.selected_memory_id:
            memory = next((item for item in self.app.ctx.state.list_memories() if item.id == self.selected_memory_id), None)
            if memory:
                self.app.ctx.state.update_memory(memory.id, enabled=not memory.enabled)
                self.refresh_memories()

    def action_send(self) -> None:
        field = self.query_one("#chat-input", Input)
        question = field.value.strip()
        if not question or self.cancel_token:
            return
        field.value = ""
        mode = str(self.query_one("#chat-mode", Select).value)
        collection = str(self.query_one("#chat-collection", Select).value)
        transcript = self.query_one(Transcript)
        if len(transcript.children) == 1 and transcript.children[0].has_class("status"):
            transcript.remove_children()
        self.query_one("#chat-sources", Markdown).update("")
        transcript.add_message("user", question)
        self._answer = ""
        self._assistant = transcript.add_message("assistant", "▍", "streaming")
        self.cancel_token = CancelToken()
        self.query_one("#stop-chat", Button).disabled = False
        self.query_one("#send-chat", Button).disabled = True
        self.query_one("#chat-status", Static).update(f"{mode.upper()} · generating…")
        self.run_chat(ChatRequest(self.conversation_id, question, mode, collection))

    def action_stop(self) -> None:
        if self.cancel_token:
            self.cancel_token.cancel()

    def _stream_event(self, event) -> None:
        if isinstance(event, StreamDelta):
            self._answer += event.text
            if self._assistant:
                self._assistant.update(f"**AutoMemory**  _streaming_\n\n{self._answer}▍")

    def _finish(self, status: str, error: Exception | None = None) -> None:
        if self._assistant:
            suffix = "" if status == "complete" else f"  _{status}_"
            self._assistant.update(f"**AutoMemory**{suffix}\n\n{self._answer or '(no content)'}")
        self.cancel_token = None
        self.query_one("#stop-chat", Button).disabled = True
        self.query_one("#send-chat", Button).disabled = False
        self.query_one("#chat-status", Static).update(f"{status} · ready")
        if error and not isinstance(error, JobCancelled):
            self.notify_error(error)

    def _show_sources(self, sources: list[dict]) -> None:
        if not sources:
            self.query_one("#chat-sources", Markdown).update("")
            return
        lines = ["**Sources**"]
        for index, item in enumerate(sources, 1):
            lines.append(f"{index}. `{item.get('document', 'document')}` · page {item.get('page', 1)} · score {item.get('score', 0):.4f}")
        self.query_one("#chat-sources", Markdown).update("\n".join(lines))

    @work(thread=True, exclusive=True, group="chat")
    def run_chat(self, request: ChatRequest) -> None:
        try:
            result = self.app.ctx.chat.stream(request, emit=lambda event: self.app.call_from_thread(self._stream_event, event), cancel=self.cancel_token)
            self._answer = result.answer
            self.app.call_from_thread(self._show_sources, result.sources)
            self.app.call_from_thread(self._finish, "complete")
        except JobCancelled as exc:
            self.app.call_from_thread(self._finish, "interrupted", exc)
        except Exception as exc:
            self.app.call_from_thread(self._finish, "error", exc)
