from typing import cast

import mlx.core as mx


def to_probs(logits: mx.array, temperature: float) -> mx.array:
    """softmax(logits / T) over the last axis; one-hot argmax when T == 0."""
    logits = logits.astype(mx.float32)
    if temperature <= 0:
        best = mx.argmax(logits, axis=-1, keepdims=True)
        return (mx.arange(logits.shape[-1]) == best).astype(mx.float32)
    return mx.softmax(logits / temperature, axis=-1)


def sample(probs: mx.array) -> mx.array:
    return mx.random.categorical(mx.log(probs))


def _pad_vocab(a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
    # Models of one family can differ in embedding padding (e.g. 151936 vs 152064).
    size = max(a.shape[-1], b.shape[-1])
    return (
        mx.pad(a, [(0, 0), (0, size - a.shape[-1])]),
        mx.pad(b, [(0, 0), (0, size - b.shape[-1])]),
    )


def verify(
    draft_tokens: mx.array, draft_probs: mx.array, target_probs: mx.array
) -> tuple[int, int]:
    """Leviathan/Chen speculative rejection sampling.

    draft_probs[i] is q(. | prefix + draft[:i]) and target_probs[i] is p(. | same);
    target_probs has one extra row for the bonus token when every draft is accepted.
    Returns (number of accepted draft tokens, the token that follows them).
    """
    n = draft_tokens.size
    if draft_probs.shape[0] != n:
        raise ValueError("draft_probs must align with draft_tokens")
    if target_probs.shape[0] != n + 1:
        raise ValueError("target_probs must have len(draft_tokens) + 1 rows")

    q, p = _pad_vocab(draft_probs, target_probs)
    rows = mx.arange(n)
    p_tok = p[rows, draft_tokens]
    q_tok = q[rows, draft_tokens]
    # u < min(1, p/q)  <=>  u * q < p, since q > 0 for a token sampled from q.
    accepted = cast(list[bool], (mx.random.uniform(shape=(n,)) * q_tok < p_tok).tolist())
    num_accepted = next((i for i, ok in enumerate(accepted) if not ok), n)

    if num_accepted == n:
        return n, cast(int, sample(p[n]).item())
    residual = mx.maximum(p[num_accepted] - q[num_accepted], 0.0)
    if cast(float, residual.sum().item()) <= 0:
        residual = p[num_accepted]
    return num_accepted, cast(int, sample(residual).item())
