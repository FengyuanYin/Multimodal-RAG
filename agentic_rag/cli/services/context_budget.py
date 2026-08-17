"""Conservative token accounting for immutable full-document prompts."""

from __future__ import annotations

import math

from ..errors import ConfigurationError
from ..models import ContextBudget


class ConservativeTokenEstimator:
    name = "conservative-unicode-v1"

    def estimate_text(self, value: str) -> int:
        ascii_count = sum(ord(char) < 128 for char in value)
        non_ascii = len(value) - ascii_count
        return max(1, math.ceil((ascii_count / 4 + non_ascii / 1.5) * 1.10))

    def estimate_messages(self, messages: list[dict]) -> int:
        return sum(self.estimate_text(str(item.get("content", ""))) + 12 for item in messages)


class ContextBudgetService:
    def __init__(self, config, estimator=None) -> None:
        self.config = config
        self.estimator = estimator or ConservativeTokenEstimator()

    def calculate(self, fixed, summary: str, events: list[dict], question: str) -> ContextBudget:
        fixed_tokens = self.estimator.estimate_messages(fixed)
        summary_tokens = self.estimator.estimate_text(summary) if summary else 0
        history_tokens = sum(self.estimator.estimate_text(item.get("content", "")) + 12 for item in events if item.get("role") != "tool")
        tool_tokens = sum(self.estimator.estimate_text(item.get("content", "")) + 12 for item in events if item.get("role") == "tool")
        question_tokens = self.estimator.estimate_text(question)
        used = fixed_tokens + summary_tokens + history_tokens + tool_tokens + question_tokens
        return ContextBudget(self.config.document_context_window_tokens, self.config.document_max_input_tokens, fixed_tokens, summary_tokens, history_tokens, tool_tokens, question_tokens, self.config.document_output_reserve_tokens, self.config.document_safety_reserve_tokens, self.estimator.name, used >= self.config.document_compaction_trigger_tokens)

    def assert_fixed_fits(self, fixed, question: str) -> None:
        required = self.estimator.estimate_messages(fixed) + self.estimator.estimate_text(question)
        if required > self.config.document_max_input_tokens:
            raise ConfigurationError(f"The complete Markdown requires about {required} input tokens, above the configured {self.config.document_max_input_tokens} limit", hint="Use a model/profile with a larger context window; AutoMemory will not truncate the document")
