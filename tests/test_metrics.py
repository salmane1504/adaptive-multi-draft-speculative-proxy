import pytest

from src.core.classifier import Domain
from src.metrics.store import MetricsStore


def record(store: MetricsStore, tps: float = 40.0):
    return store.record(
        domain=Domain.CODE,
        draft_model="draft",
        k=4,
        acceptance_rate=0.7,
        tokens=10,
        tokens_per_second=tps,
        dispatch_latency_ms=12.0,
    )


def test_ring_buffer_keeps_most_recent_records():
    store = MetricsStore(capacity=3)
    for tps in [1.0, 2.0, 3.0, 4.0]:
        record(store, tps)

    assert [m.tokens_per_second for m in store.recent(10)] == [2.0, 3.0, 4.0]
    assert [m.tokens_per_second for m in store.recent(2)] == [3.0, 4.0]


def test_speedup_is_relative_to_baseline():
    store = MetricsStore()
    assert record(store).speedup is None

    store.update_baseline(20.0)
    metrics = record(store)

    assert metrics.baseline_tokens_per_second == 20.0
    assert metrics.speedup == 2.0


def test_baseline_is_smoothed_and_ignores_empty_runs():
    store = MetricsStore(smoothing=0.5)
    store.update_baseline(20.0)
    store.update_baseline(0.0)
    store.update_baseline(30.0)

    assert store.baseline_tps == pytest.approx(25.0)
