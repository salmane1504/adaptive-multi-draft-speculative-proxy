import random

import pytest

from src.core.sampler import tempered_dist, verify


def test_tempered_dist_greedy_is_one_hot():
    assert tempered_dist({"a": -0.1, "b": -2.0}, 0.0) == {"a": 1.0}


def test_tempered_dist_normalises():
    dist = tempered_dist({"a": -0.5, "b": -1.0, "c": -3.0}, 0.7)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["a"] > dist["b"] > dist["c"]


def test_verify_accepts_all_and_samples_bonus():
    rng = random.Random(0)
    result = verify(
        ["a", "b"],
        [{"a": 1.0}, {"b": 1.0}],
        [{"a": 1.0}, {"b": 1.0}, {"c": 1.0}],
        rng,
    )
    assert result.tokens == ["a", "b", "c"]
    assert result.num_accepted == 2


def test_verify_rejects_and_corrects_from_residual():
    rng = random.Random(0)
    result = verify(
        ["a", "x", "b"],
        [{"a": 1.0}, {"x": 1.0}, {"b": 1.0}],
        [{"a": 1.0}, {"y": 1.0}, {"b": 1.0}, {"c": 1.0}],
        rng,
    )
    assert result.tokens == ["a", "y"]
    assert result.num_accepted == 1


def test_verify_preserves_target_distribution():
    # Output marginal must equal p regardless of q (Leviathan et al., Thm. 1).
    p = {"a": 0.2, "b": 0.5, "c": 0.3}
    q = {"a": 0.6, "b": 0.3, "c": 0.1}
    rng = random.Random(1)
    counts = {"a": 0, "b": 0, "c": 0}
    n = 40_000
    for _ in range(n):
        draft_tok = rng.choices(list(q), weights=list(q.values()))[0]
        result = verify([draft_tok], [q], [p, p], rng)
        counts[result.tokens[0]] += 1
    for tok, prob in p.items():
        assert counts[tok] / n == pytest.approx(prob, abs=0.01)
