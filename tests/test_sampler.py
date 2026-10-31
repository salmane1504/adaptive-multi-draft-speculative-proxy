from typing import cast

import mlx.core as mx
import pytest

from src.core.sampler import to_probs, verify


def one_hot(*tokens: int, vocab: int = 5) -> mx.array:
    return (mx.arange(vocab) == mx.array(tokens)[:, None]).astype(mx.float32)


def test_to_probs_greedy_is_one_hot():
    assert to_probs(mx.array([[0.1, 2.0, -1.0]]), 0.0).tolist() == [[0.0, 1.0, 0.0]]


def test_to_probs_normalises():
    probs = cast(list[float], to_probs(mx.array([-0.5, -1.0, -3.0]), 0.7).tolist())
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]


def test_verify_accepts_all_and_samples_bonus():
    result = verify(mx.array([0, 1]), one_hot(0, 1), one_hot(0, 1, 2))
    assert result == (2, 2)


def test_verify_rejects_and_corrects_from_residual():
    result = verify(mx.array([0, 3, 1]), one_hot(0, 3, 1), one_hot(0, 4, 1, 2))
    assert result == (1, 4)


def test_verify_pads_mismatched_vocab_sizes():
    result = verify(mx.array([1]), one_hot(1, vocab=3), one_hot(1, 4, vocab=6))
    assert result == (1, 4)


def test_verify_preserves_target_distribution():
    # Output marginal must equal p regardless of q (Leviathan et al., Thm. 1).
    mx.random.seed(1)
    p = mx.array([[0.2, 0.5, 0.3]] * 2)
    q = mx.array([[0.6, 0.3, 0.1]])
    n = 5_000
    counts = [0, 0, 0]
    drafts = cast(list[int], mx.random.categorical(mx.log(q[0]), num_samples=n).tolist())
    for draft_tok in drafts:
        num_accepted, bonus = verify(mx.array([draft_tok]), q, p)
        counts[draft_tok if num_accepted else bonus] += 1
    for tok, prob in enumerate([0.2, 0.5, 0.3]):
        assert counts[tok] / n == pytest.approx(prob, abs=0.03)
