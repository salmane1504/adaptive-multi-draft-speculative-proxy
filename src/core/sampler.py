import math
import random
from dataclasses import dataclass

TokenDist = dict[str, float]


def tempered_dist(logprobs: dict[str, float], temperature: float) -> TokenDist:
    """Softmax(logprob / T) renormalised over the observed top-N support."""
    if not logprobs:
        return {}
    if temperature <= 0:
        best = max(logprobs, key=logprobs.__getitem__)
        return {best: 1.0}
    scaled = {tok: lp / temperature for tok, lp in logprobs.items()}
    peak = max(scaled.values())
    weights = {tok: math.exp(s - peak) for tok, s in scaled.items()}
    total = sum(weights.values())
    return {tok: w / total for tok, w in weights.items()}


def sample(dist: TokenDist, rng: random.Random) -> str:
    total = sum(dist.values())
    if total <= 0:
        raise ValueError("cannot sample from an empty distribution")
    threshold = rng.random() * total
    cumulative = 0.0
    for tok, prob in dist.items():
        cumulative += prob
        if cumulative >= threshold:
            return tok
    return next(reversed(dist))


@dataclass(frozen=True)
class VerificationResult:
    tokens: list[str]  # accepted draft prefix + one target-sampled token
    num_accepted: int


def verify(
    draft_tokens: list[str],
    draft_dists: list[TokenDist],
    target_dists: list[TokenDist],
    rng: random.Random,
) -> VerificationResult:
    """Leviathan/Chen speculative rejection sampling.

    target_dists[i] is p(. | prefix + draft[:i]); it has one extra entry used
    to sample the bonus token when every draft token is accepted.
    """
    if len(draft_dists) != len(draft_tokens):
        raise ValueError("draft_dists must align with draft_tokens")
    if len(target_dists) != len(draft_tokens) + 1:
        raise ValueError("target_dists must have len(draft_tokens) + 1 entries")

    out: list[str] = []
    for i, tok in enumerate(draft_tokens):
        p = target_dists[i].get(tok, 0.0)
        q = draft_dists[i].get(tok, 0.0)
        if q > 0 and rng.random() < min(1.0, p / q):
            out.append(tok)
            continue

        residual = {
            t: max(0.0, pt - draft_dists[i].get(t, 0.0))
            for t, pt in target_dists[i].items()
        }
        if sum(residual.values()) <= 0:
            residual = target_dists[i]
        out.append(sample(residual, rng))
        return VerificationResult(tokens=out, num_accepted=i)

    out.append(sample(target_dists[-1], rng))
    return VerificationResult(tokens=out, num_accepted=len(draft_tokens))
