import asyncio

from src.clients.mlx_client import Completion
from src.core.engine import SpeculativeEngine
from src.core.router import DynamicRouter

PROMPT = {"messages": [{"role": "user", "content": "Explain the seasons."}]}
TEXT = "The Earth is tilted."


class FakeClient:
    """Accepts 6 of 8 drafted tokens whenever a draft model is supplied."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, int, int]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        draft_model: str | None,
        num_draft_tokens: int,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        self.calls.append((draft_model, num_draft_tokens, max_tokens))
        if draft_model is None:
            return Completion(TEXT, "stop", 5, 0, 0, 0)
        return Completion(TEXT, "stop", 5, 6, 8, 2)


def test_speculative_request_uses_routed_draft_and_records_feedback():
    router = DynamicRouter(target_model="target")
    client = FakeClient()
    engine = SpeculativeEngine(client, router)

    result = asyncio.run(engine.generate({**PROMPT, "temperature": 0.0, "max_tokens": 32}))

    route = router.route(PROMPT)
    assert client.calls == [("mlx-community/Qwen2.5-0.5B-Instruct-8bit", 3, 32)]
    assert result.text == TEXT
    assert result.draft_model == route.draft_model
    assert result.acceptance_rate == 0.75
    assert list(router.stats[result.domain].history) == [0.75]


def test_bypass_runs_target_only_and_skips_feedback():
    router = DynamicRouter(target_model="target")
    router.record_execution_feedback(router.route(PROMPT).domain, 0, 10)
    client = FakeClient()
    engine = SpeculativeEngine(client, router)

    result = asyncio.run(engine.generate({**PROMPT, "max_tokens": 32}))

    assert result.text == TEXT
    assert result.draft_model is None
    assert client.calls == [(None, 0, 32)]
    assert list(router.stats[result.domain].history) == [0.0]
