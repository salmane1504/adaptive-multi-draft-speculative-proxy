import mlx.core as mx
import pytest

from src.core.decoding import speculative_decode, target_decode

VOCAB = 32
PROMPT = [1, 2, 3, 4, 5]


class ToyLM:
    """Logits depend on token and absolute position, so a mis-rewound KV cache
    shifts positions and changes the output."""

    def __init__(self, table: mx.array):
        self.table = table
        self.layers = [None]

    def __call__(self, tokens: mx.array, cache: list) -> mx.array:
        offset, length = cache[0].offset, tokens.shape[1]
        kv = mx.zeros((1, 1, length, 1))
        cache[0].update_and_fetch(kv, kv)
        return self.table[(tokens + mx.arange(offset, offset + length)) % VOCAB]


mx.random.seed(0)
TARGET = ToyLM(mx.random.normal((VOCAB, VOCAB)) * 3)
DRAFT = ToyLM(TARGET.table + mx.random.normal((VOCAB, VOCAB)) * 2)


def reference(max_tokens: int = 40) -> list[int]:
    return target_decode(
        TARGET, PROMPT, max_tokens=max_tokens, temperature=0.0, eos_ids=set()
    ).tokens


@pytest.mark.parametrize("k", [1, 3, 8])
def test_greedy_speculative_matches_target_only_decoding(k):
    out = speculative_decode(
        TARGET, DRAFT, PROMPT, num_draft_tokens=k, max_tokens=40, temperature=0.0, eos_ids=set()
    )

    assert out.tokens == reference()
    assert out.finish_reason == "length"
    assert 0 < out.accepted_tokens < out.drafted_tokens


def test_identical_draft_accepts_every_token():
    out = speculative_decode(
        TARGET, TARGET, PROMPT, num_draft_tokens=4, max_tokens=40, temperature=0.0, eos_ids=set()
    )

    assert out.tokens == reference()
    assert out.accepted_tokens == out.drafted_tokens


def test_stops_before_eos():
    ref = reference()
    eos = ref[10]
    expected = ref[: ref.index(eos)]

    spec = speculative_decode(
        TARGET, DRAFT, PROMPT, num_draft_tokens=3, max_tokens=40, temperature=0.0, eos_ids={eos}
    )
    plain = target_decode(TARGET, PROMPT, max_tokens=40, temperature=0.0, eos_ids={eos})

    assert spec.tokens == plain.tokens == expected
    assert spec.finish_reason == plain.finish_reason == "stop"


def test_sampling_respects_max_tokens():
    out = speculative_decode(
        TARGET, DRAFT, PROMPT, num_draft_tokens=5, max_tokens=30, temperature=0.8, eos_ids=set()
    )

    assert len(out.tokens) == 30
    assert all(0 <= t < VOCAB for t in out.tokens)
    assert out.finish_reason == "length"
