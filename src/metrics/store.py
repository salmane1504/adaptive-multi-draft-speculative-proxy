import time
from collections import deque

from pydantic import BaseModel

from src.core.classifier import Domain


class RequestMetrics(BaseModel):
    timestamp: float
    domain: Domain
    draft_model: str | None
    k: int
    acceptance_rate: float | None
    tokens: int
    tokens_per_second: float
    baseline_tokens_per_second: float | None
    speedup: float | None
    dispatch_latency_ms: float


class MetricsStore:
    """In-memory ring buffer of per-request metrics plus a target-only tok/s baseline."""

    def __init__(self, capacity: int = 1000, smoothing: float = 0.3):
        self._records: deque[RequestMetrics] = deque(maxlen=capacity)
        self._smoothing = smoothing
        self.baseline_tps: float | None = None

    def update_baseline(self, tokens_per_second: float) -> None:
        if tokens_per_second <= 0:
            return
        if self.baseline_tps is None:
            self.baseline_tps = tokens_per_second
        else:
            a = self._smoothing
            self.baseline_tps = a * tokens_per_second + (1 - a) * self.baseline_tps

    def record(
        self,
        *,
        domain: Domain,
        draft_model: str | None,
        k: int,
        acceptance_rate: float | None,
        tokens: int,
        tokens_per_second: float,
        dispatch_latency_ms: float,
    ) -> RequestMetrics:
        baseline = self.baseline_tps
        metrics = RequestMetrics(
            timestamp=time.time(),
            domain=domain,
            draft_model=draft_model,
            k=k,
            acceptance_rate=acceptance_rate,
            tokens=tokens,
            tokens_per_second=tokens_per_second,
            baseline_tokens_per_second=baseline,
            speedup=tokens_per_second / baseline if baseline and tokens_per_second else None,
            dispatch_latency_ms=dispatch_latency_ms,
        )
        self._records.append(metrics)
        return metrics

    def recent(self, limit: int) -> list[RequestMetrics]:
        return list(self._records)[-limit:]
