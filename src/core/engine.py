import logging
from dataclasses import dataclass
from typing import Any, Protocol

from src.clients.mlx_client import Completion
from src.core.classifier import Domain
from src.core.decoding import FinishReason
from src.core.router import DynamicRouter
from src.metrics.store import MetricsStore, RequestMetrics

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        draft_model: str | None,
        num_draft_tokens: int,
        max_tokens: int,
        temperature: float,
    ) -> Completion: ...


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: FinishReason
    domain: Domain
    draft_model: str | None
    accepted_tokens: int
    drafted_tokens: int
    cycles: int
    metrics: RequestMetrics

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0


CALIBRATION_PROMPT = [{"role": "user", "content": "Count from 1 to 100, comma separated."}]


class SpeculativeEngine:
    def __init__(
        self,
        client: LLMClient,
        router: DynamicRouter,
        metrics: MetricsStore | None = None,
    ):
        self.client = client
        self.router = router
        self.metrics = metrics or MetricsStore()

    async def calibrate(self) -> None:
        """Warm up every draft, then measure target-only tok/s as the speedup baseline."""
        for draft in sorted(set(self.router.draft_registry.values())):
            await self.client.generate(
                CALIBRATION_PROMPT,
                draft_model=draft,
                num_draft_tokens=2,
                max_tokens=16,
                temperature=0.0,
            )
        completion = await self.client.generate(
            CALIBRATION_PROMPT,
            draft_model=None,
            num_draft_tokens=0,
            max_tokens=64,
            temperature=0.0,
        )
        self.metrics.update_baseline(completion.tokens_per_second)
        logger.info("Baseline (target only): %.1f tok/s", completion.tokens_per_second)

    async def generate(self, payload: dict[str, Any]) -> GenerationResult:
        route = self.router.route(payload)
        completion = await self.client.generate(
            payload.get("messages", []),
            draft_model=route.draft_model,
            num_draft_tokens=route.k_drafts,
            max_tokens=int(payload.get("max_tokens", 256)),
            temperature=float(payload.get("temperature", 0.7)),
        )

        speculative = route.draft_model is not None
        if speculative:
            self.router.record_execution_feedback(
                route.domain, completion.accepted_tokens, completion.drafted_tokens
            )
        acceptance = (
            completion.accepted_tokens / completion.drafted_tokens
            if completion.drafted_tokens
            else None
        )
        metrics = self.metrics.record(
            domain=route.domain,
            draft_model=route.draft_model,
            k=route.k_drafts,
            acceptance_rate=acceptance,
            tokens=completion.completion_tokens,
            tokens_per_second=completion.tokens_per_second,
            dispatch_latency_ms=completion.dispatch_latency_ms,
        )
        if not speculative and completion.completion_tokens > 1:
            self.metrics.update_baseline(completion.tokens_per_second)

        logger.info(
            "domain=%s draft=%s k=%d tokens=%d accepted=%d/%d %.1f tok/s speedup=%s",
            route.domain.value,
            route.draft_model or "none",
            route.k_drafts,
            completion.completion_tokens,
            completion.accepted_tokens,
            completion.drafted_tokens,
            completion.tokens_per_second,
            f"{metrics.speedup:.2f}x" if metrics.speedup else "n/a",
        )
        return GenerationResult(
            text=completion.text,
            finish_reason=completion.finish_reason,
            domain=route.domain,
            draft_model=route.draft_model,
            accepted_tokens=completion.accepted_tokens,
            drafted_tokens=completion.drafted_tokens,
            cycles=completion.cycles,
            metrics=metrics,
        )
