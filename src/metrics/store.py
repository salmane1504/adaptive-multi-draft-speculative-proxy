import logging
import time
from collections import deque
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.core.classifier import Domain

logger = logging.getLogger(__name__)


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
    """Ring buffer of per-request metrics plus a target-only tok/s baseline.

    With `path`, each record is appended to a JSONL file and the buffer is reloaded from it.
    """

    def __init__(self, capacity: int = 1000, smoothing: float = 0.3, path: Path | None = None):
        self._records: deque[RequestMetrics] = deque(maxlen=capacity)
        self._smoothing = smoothing
        self._path = path
        self.baseline_tps: float | None = None
        if path is not None:
            self._load(path)

    def _load(self, path: Path) -> None:
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    self._records.append(RequestMetrics.model_validate_json(line))
                except ValidationError:
                    logger.warning("Skipping malformed metrics line in %s", path)

    def _append(self, metrics: RequestMetrics) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(metrics.model_dump_json() + "\n")
        except OSError:
            logger.warning("Could not write metrics to %s", self._path, exc_info=True)

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
        self._append(metrics)
        return metrics

    def recent(self, limit: int) -> list[RequestMetrics]:
        return list(self._records)[-limit:]
