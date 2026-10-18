import asyncio

from src.clients.ollama_client import Completion, TokenLogprob
from src.core.engine import SpeculativeEngine, format_chatml
from src.core.router import DynamicRouter

PROMPT = {"messages": [{"role": "user", "content": "Explain the seasons."}]}
TARGET_TEXT = ["The", " Earth", " is", " tilted", ".", "<|im_end|>"]


class FakeClient:
    """Target deterministically emits TARGET_TEXT; draft is wrong only at index 1."""

    def __init__(self) -> None:
        self.base = format_chatml(PROMPT["messages"])
        self.calls: list[tuple[str, int]] = []

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        num_predict: int,
        temperature: float,
        logprobs: bool = True,
        top_logprobs: int = 20,
        seed: int | None = None,
    ) -> Completion:
        self.calls.append((model, num_predict))
        pos = self._position(prompt)
        if model == "target":
            if not logprobs:
                return Completion("".join(TARGET_TEXT[:-1]), [], "stop")
            tok = TARGET_TEXT[pos]
            return Completion(tok, [TokenLogprob(tok, 0.0, {tok: 0.0})], "length")

        draft = list(TARGET_TEXT)
        draft[1] = " Venus"
        toks = draft[pos : pos + num_predict]
        return Completion("".join(toks), [TokenLogprob(t, 0.0, {t: 0.0}) for t in toks], "length")

    def _position(self, prompt: str) -> int:
        rest = prompt[len(self.base) :]
        pos = 0
        while rest:
            rest = rest[len(TARGET_TEXT[pos]) :]
            pos += 1
        return pos


def test_speculative_generation_matches_target_greedy_output():
    router = DynamicRouter(target_model="target")
    engine = SpeculativeEngine(FakeClient(), router, seed=0)

    result = asyncio.run(engine.generate({**PROMPT, "temperature": 0.0, "max_tokens": 32}))

    assert result.text == "The Earth is tilted."
    assert result.drafted_tokens > 0
    assert 0 < result.accepted_tokens < result.drafted_tokens
    assert router.stats[result.domain].history


def test_bypass_uses_single_target_call():
    router = DynamicRouter(target_model="target")
    router.record_execution_feedback(router.route(PROMPT).domain, 0, 10)
    client = FakeClient()
    engine = SpeculativeEngine(client, router)

    result = asyncio.run(engine.generate({**PROMPT, "max_tokens": 32}))

    assert result.text == "The Earth is tilted."
    assert result.draft_model is None
    assert client.calls == [("target", 32)]
