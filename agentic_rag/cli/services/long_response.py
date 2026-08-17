"""Store long answers as Markdown and retain an explicitly incomplete preview."""

from __future__ import annotations


class LongResponseService:
    def __init__(self, config, files, estimator) -> None:
        self.config, self.files, self.estimator = config, files, estimator

    def finalize(self, workspace_id: str, answer: str) -> tuple[str, str | None, dict]:
        estimate = self.estimator.estimate_text(answer)
        if estimate < self.config.document_long_answer_tokens:
            return answer, None, {"complete": True, "token_estimate": estimate}
        record = self.files.create_markdown(workspace_id, "long_answer", "full-answer.md", answer, "Complete main LLM answer")
        head_chars, tail_chars = self.config.document_preview_head_tokens * 3, self.config.document_preview_tail_tokens * 3
        preview = f'<AUTOMEMORY_ANSWER_PREVIEW complete="false" file_id="{record["id"]}" checksum="{record["checksum"]}" total_chars="{len(answer)}">\n[BEGINNING OF ANSWER]\n{answer[:head_chars]}\n[MIDDLE OMITTED — READ THE FILE FOR COMPLETE CONTENT]\n{answer[-tail_chars:]}\n[END OF ANSWER]\n</AUTOMEMORY_ANSWER_PREVIEW>'
        return preview, record["id"], {"complete": False, "token_estimate": estimate, "file_id": record["id"]}
