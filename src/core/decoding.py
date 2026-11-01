import time
from dataclasses import dataclass
from typing import Any, Literal, cast

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from src.core.sampler import sample, to_probs, verify

PREFILL_STEP = 512

FinishReason = Literal["stop", "length"]


@dataclass(frozen=True)
class DecodeOutput:
    tokens: list[int]
    finish_reason: FinishReason
    accepted_tokens: int = 0
    drafted_tokens: int = 0
    cycles: int = 0
    # Generation phase only (prompt prefill excluded).
    tokens_per_second: float = 0.0
    # Mean wall time of one target forward pass, from dispatch to result.
    dispatch_latency_ms: float = 0.0


def target_decode(
    model: Any,
    prompt: list[int],
    *,
    max_tokens: int,
    temperature: float,
    eos_ids: set[int],
) -> DecodeOutput:
    """Plain autoregressive decoding, used when speculation is bypassed."""
    tokens: list[int] = []
    finish: FinishReason = "length"
    steps = generate_step(
        mx.array(prompt, mx.uint32),
        model,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
    )
    first_at = 0.0
    passes = 0
    for token, _ in steps:
        if passes == 0:
            first_at = time.perf_counter()
        passes += 1
        if token in eos_ids:
            finish = "stop"
            break
        tokens.append(cast(int, token))

    # The first token comes out of prefill, so time is measured from it.
    elapsed = time.perf_counter() - first_at
    timed = passes - 1
    return DecodeOutput(
        tokens,
        finish,
        tokens_per_second=timed / elapsed if timed > 0 else 0.0,
        dispatch_latency_ms=elapsed / timed * 1000 if timed > 0 else 0.0,
    )


def speculative_decode(
    target: Any,
    draft: Any,
    prompt: list[int],
    *,
    num_draft_tokens: int,
    max_tokens: int,
    temperature: float,
    eos_ids: set[int],
) -> DecodeOutput:
    """Draft K tokens, score all K+1 positions in ONE target forward pass, then
    rejection-sample and rewind both KV caches past the first rejected token."""
    target_cache = make_prompt_cache(target)
    draft_cache = make_prompt_cache(draft)
    prompt_arr = mx.array(prompt, mx.uint32)
    target_in = _prefill(target, target_cache, prompt_arr)
    draft_in = _prefill(draft, draft_cache, prompt_arr)

    tokens: list[int] = []
    finish: FinishReason = "length"
    accepted = drafted = cycles = 0
    verify_seconds = 0.0
    decode_start = time.perf_counter()
    while True:
        n = min(num_draft_tokens, max_tokens - len(tokens))
        draft_tokens, draft_probs = _draft(draft, draft_cache, draft_in, n, temperature)
        mx.eval(draft_tokens, draft_probs)

        dispatched = time.perf_counter()
        logits = target(mx.concatenate([target_in, draft_tokens])[None], cache=target_cache)
        target_probs = to_probs(logits[0, -(n + 1) :], temperature)
        num_accepted, next_token = verify(draft_tokens, draft_probs, target_probs)
        verify_seconds += time.perf_counter() - dispatched

        drafted_list = cast(list[int], draft_tokens.tolist())
        accepted += num_accepted
        drafted += n
        cycles += 1

        done = False
        for token in [*drafted_list[:num_accepted], next_token]:
            if token in eos_ids:
                finish, done = "stop", True
                break
            tokens.append(token)
            if len(tokens) == max_tokens:
                done = True
                break
        if done:
            break

        # Target saw [last, d_0..d_{n-1}]; draft saw [last, d_0..d_{n-2}].
        trim_prompt_cache(target_cache, n - num_accepted)
        target_in = mx.array([next_token], mx.uint32)
        if num_accepted == n:
            draft_in = mx.array([drafted_list[-1], next_token], mx.uint32)
        else:
            trim_prompt_cache(draft_cache, n - num_accepted - 1)
            draft_in = mx.array([next_token], mx.uint32)

    elapsed = time.perf_counter() - decode_start
    return DecodeOutput(
        tokens,
        finish,
        accepted,
        drafted,
        cycles,
        tokens_per_second=len(tokens) / elapsed if elapsed > 0 else 0.0,
        dispatch_latency_ms=verify_seconds / cycles * 1000,
    )


def _prefill(model: Any, cache: list[Any], tokens: mx.array) -> mx.array:
    """Feed all but the last token into the cache and return that last token."""
    while tokens.size > 1:
        step = min(PREFILL_STEP, tokens.size - 1)
        model(tokens[:step][None], cache=cache)
        mx.eval([c.state for c in cache])
        tokens = tokens[step:]
        mx.clear_cache()
    return tokens


def _draft(
    model: Any, cache: list[Any], tokens: mx.array, n: int, temperature: float
) -> tuple[mx.array, mx.array]:
    drafted, dists = [], []
    for _ in range(n):
        probs = to_probs(model(tokens[None], cache=cache)[0, -1], temperature)
        tokens = sample(probs).reshape(1).astype(mx.uint32)
        mx.async_eval(tokens)
        drafted.append(tokens)
        dists.append(probs)
    return mx.concatenate(drafted), mx.stack(dists)
