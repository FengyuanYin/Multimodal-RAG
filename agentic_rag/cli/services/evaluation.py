"""Deterministic retrieval evaluation and atomic export."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile
from time import perf_counter
import uuid

from ..cancellation import CancellationToken
from ..errors import CancelledError, ConfigurationError
from ..models import EventKind, OutputEvent
from ..security import ensure_within


def evaluate_hits(hits, expected: list, expected_media: list, top_k: int) -> dict:
    relevant, media_relevant = {str(item) for item in expected}, {str(item) for item in expected_media}
    identifiers = [{hit.target_id, hit.document_id} for hit in hits[:top_k]]
    media_found = {str(ref.get("media_id") or ref.get("id") or "") for hit in hits[:top_k] for ref in hit.media_refs if isinstance(ref, dict)}
    flags = [bool(item & relevant) for item in identifiers]
    hit_count = sum(flags)
    first = next((index for index, flag in enumerate(flags, 1) if flag), None)
    dcg = sum(1 / math.log2(index + 1) for index, flag in enumerate(flags, 1) if flag)
    ideal = sum(1 / math.log2(index + 1) for index in range(1, min(len(relevant), top_k) + 1))
    return {
        "precision_at_k": hit_count / max(1, top_k),
        "recall_at_k": hit_count / len(relevant) if relevant else None,
        "mrr": 1 / first if first else 0.0,
        "ndcg_at_k": dcg / ideal if ideal else None,
        "media_recall_at_k": len(media_found & media_relevant) / len(media_relevant) if media_relevant else None,
    }


class EvaluationService:
    def __init__(self, retriever, state, exports_dir: Path) -> None:
        self.retriever, self.state, self.exports_dir = retriever, state, exports_dir

    @staticmethod
    def load_dataset(path: Path) -> list[dict]:
        try:
            value = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Evaluation dataset is invalid: {exc}") from exc
        if isinstance(value, dict):
            value = value.get("cases") or value.get("items")
        if not isinstance(value, list) or not value:
            raise ConfigurationError("Evaluation dataset must contain a non-empty list")
        if any(not isinstance(item, dict) or not str(item.get("query") or "").strip() for item in value):
            raise ConfigurationError("Each evaluation case must include a query")
        return value

    def run(self, path: Path, mode: str, top_k: int, scope: str, output, cancel: CancellationToken) -> dict:
        dataset = self.load_dataset(path)
        run_id = f"eval_{uuid.uuid4().hex}"
        self.state.create_evaluation(run_id, str(path), {"mode": mode, "top_k": top_k, "scope": scope})
        cases = []
        try:
            for index, item in enumerate(dataset, 1):
                cancel.checkpoint()
                output.emit(OutputEvent(EventKind.PROGRESS, text=str(item["query"]), task_id=run_id, phase="retrieve", completed=index - 1, total=len(dataset)))
                started = perf_counter()
                result = self.retriever.search(str(item["query"]), scope, mode, top_k, cancel)
                metrics = evaluate_hits(result.hits, item.get("expected", []), item.get("expected_media", []), top_k)
                cases.append({"id": item.get("id", f"case_{index}"), "query": item["query"], **metrics, "latency_ms": round((perf_counter() - started) * 1000, 3)})
            fields = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "media_recall_at_k", "latency_ms"]
            summary = {name: round(mean(values), 6) if (values := [case[name] for case in cases if case.get(name) is not None]) else None for name in fields}
            self.state.update_evaluation(run_id, "success", summary)
            return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "mode": mode, "top_k": top_k, "count": len(cases), "summary": summary, "cases": cases}
        except CancelledError:
            self.state.update_evaluation(run_id, "cancelled")
            raise
        except Exception:
            self.state.update_evaluation(run_id, "error")
            raise

    def export(self, result: dict, filename: str | None = None) -> Path:
        destination = ensure_within(self.exports_dir, self.exports_dir / (filename or f"evaluation-{result['run_id']}.json"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".automemory-eval-", suffix=".json", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination
