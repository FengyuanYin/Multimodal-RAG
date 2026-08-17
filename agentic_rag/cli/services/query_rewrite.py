"""Cloud query rewriting with a safe original-query fallback."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import CancelledError


@dataclass(slots=True)
class QueryRewriteResult:
    queries: list[str]
    degraded: list[dict] = field(default_factory=list)


class QueryRewriteService:
    def __init__(self, llm_client=None) -> None:
        self.llm_client = llm_client

    def rewrite(self, question: str, limit: int, cancel) -> QueryRewriteResult:
        original = question.strip()
        if not self.llm_client or limit <= 1:
            return QueryRewriteResult([original], [] if self.llm_client else [{"stage":"query_rewrite","reason":"cloud LLM not configured"}])
        try:
            data = self.llm_client.complete_json([
                {"role":"system","content":"Rewrite the question into complementary retrieval queries. Return JSON: {\"queries\":[...]}. Preserve names and technical terms."},
                {"role":"user","content":original},
            ], cancel, max_tokens=500)
            output = [original]
            for value in data.get("queries") or []:
                item = str(value).strip()[:500]
                if item and item.casefold() not in {query.casefold() for query in output}:
                    output.append(item)
                if len(output) >= limit:
                    break
            return QueryRewriteResult(output)
        except CancelledError:
            raise
        except Exception as exc:
            return QueryRewriteResult([original], [{"stage":"query_rewrite","reason":type(exc).__name__}])
