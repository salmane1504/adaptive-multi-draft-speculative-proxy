import asyncio
import random
from dataclasses import dataclass
from typing import Any, Protocol

from src.clients.ollama_client import Completion
from src.core.classifier import Domain
from src.core.router import DynamicRouter
from src.core.sampler import TokenDist, tempered_dist, verify

EOS_TOKEN = "<|im_end|>"
STOP_TOKENS = frozenset({EOS_TOKEN, "<|endoftext|>"})


class LLMClient(Protocol):
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
    ) -> Completion: ...


def format_chatml(messages: list[dict[str, str]]) -> str:
    """Qwen2.5 chat template, applied manually because requests use raw mode."""
    parts = [
        f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>\n"
        for m in messages
    ]
    return "".join(parts) + "<|im_start|>assistant\n"


@dataclass(frozen=True)
class GenerationResult:
    text: str
    domain: Domain
    draft_model: str | None
    accepted_tokens: int
    drafted_tokens: int
    cycles: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0


class SpeculativeEngine:
    def __init__(
        self,
        client: LLMClient,
        router: DynamicRouter,
        *,
        top_logprobs: int = 20,
        seed: int | None = None,
    ):
        self.client = client
        self.router = router
        self.top_logprobs = top_logprobs
        self.rng = random.Random(seed)

    async def generate(self, payload: dict[str, Any]) -> GenerationResult:
        route = self.router.route(payload)
        prompt = format_chatml(payload.get("messages", []))
        max_tokens = int(payload.get("max_tokens", 256))
        temperature = float(payload.get("temperature", 0.7))

        if route.bypass_speculation or not route.draft_model or route.k_drafts <= 0:
            completion = await self.client.generate(
                route.target_model,
                prompt,
                num_predict=max_tokens,
                temperature=temperature,
                logprobs=False,
            )
            return GenerationResult(
                text=completion.text,
                domain=route.domain,
                draft_model=None,
                accepted_tokens=0,
                drafted_tokens=0,
                cycles=0,
            )

        generated: list[str] = []
        accepted = drafted = cycles = 0
        finished = False

        while not finished and len(generated) < max_tokens:
            prefix = prompt + "".join(generated)
            k = min(route.k_drafts, max_tokens - len(generated))

            draft = await self.client.generate(
                route.draft_model,
                prefix,
                num_predict=k,
                temperature=temperature,
                top_logprobs=self.top_logprobs,
            )
            draft_entries = draft.tokens
            stop_at = next(
                (i for i, t in enumerate(draft_entries) if t.token in STOP_TOKENS), None
            )
            if stop_at is not None:
                # Let the target decide EOS at that position instead of verifying past it.
                draft_entries = draft_entries[:stop_at]
            draft_tokens = [t.token for t in draft_entries]
            draft_dists = [tempered_dist(t.top_logprobs, temperature) for t in draft_entries]
            target_dists = await self._target_dists(
                route.target_model, prefix, draft_tokens, temperature
            )

            result = verify(draft_tokens, draft_dists, target_dists, self.rng)
            accepted += result.num_accepted
            drafted += len(draft_tokens)
            cycles += 1

            for tok in result.tokens:
                if tok in STOP_TOKENS:
                    finished = True
                    break
                generated.append(tok)

        self.router.record_execution_feedback(route.domain, accepted, drafted)
        return GenerationResult(
            text="".join(generated[:max_tokens]),
            domain=route.domain,
            draft_model=route.draft_model,
            accepted_tokens=accepted,
            drafted_tokens=drafted,
            cycles=cycles,
        )

    async def _target_dists(
        self, model: str, prefix: str, draft_tokens: list[str], temperature: float
    ) -> list[TokenDist]:
        """Target next-token distributions at each of the K+1 draft positions.

        Ollama cannot score supplied tokens, so the K+1 single-token requests are
        issued concurrently and batched by the runner (needs OLLAMA_NUM_PARALLEL > K).
        """
        prompts = [prefix + "".join(draft_tokens[:i]) for i in range(len(draft_tokens) + 1)]
        completions = await asyncio.gather(
            *(
                self.client.generate(
                    model,
                    p,
                    num_predict=1,
                    temperature=0.0,
                    top_logprobs=self.top_logprobs,
                )
                for p in prompts
            )
        )
        # An empty completion means the target emitted EOS at that position.
        return [
            tempered_dist(c.tokens[0].top_logprobs, temperature)
            if c.tokens
            else {EOS_TOKEN: 1.0}
            for c in completions
        ]
