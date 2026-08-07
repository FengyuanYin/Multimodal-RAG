"""可解释检索的数据结构。

排名分数与回答置信度刻意分离：RRF/重排分数只描述排序，不能当概率使用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


@dataclass
class DegradedChannel:
    channel: str
    code: str
    reason: str
    recoverable: bool = True


@dataclass
class RetrievalCandidate:
    id: str
    document_id: str
    page: int
    modality: str
    content: str
    media_refs: List[dict] = field(default_factory=list)
    channel_scores: Dict[str, float] = field(default_factory=dict)
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    request_id: str = field(default_factory=lambda: f"ret_{uuid.uuid4().hex[:16]}")
    requested_mode: str = "hybrid"
    active_channels: List[str] = field(default_factory=list)
    degraded_channels: List[DegradedChannel] = field(default_factory=list)
    candidate_counts: Dict[str, int] = field(default_factory=dict)
    stage_latency_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter, repr=False)

    def activate(self, channel: str, count: int = 0) -> None:
        if channel not in self.active_channels:
            self.active_channels.append(channel)
        self.candidate_counts[channel] = count

    def degrade(self, channel: str, code: str, reason: str, recoverable: bool = True) -> None:
        self.degraded_channels.append(DegradedChannel(channel, code, reason, recoverable))

    def record_latency(self, stage: str, started_at: float) -> None:
        self.stage_latency_ms[stage] = round((time.perf_counter() - started_at) * 1000, 3)

    def finish(self) -> None:
        self.stage_latency_ms["total"] = round((time.perf_counter() - self.started_at) * 1000, 3)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("started_at", None)
        return data
