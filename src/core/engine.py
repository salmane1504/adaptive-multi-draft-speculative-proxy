import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from src.clients.mlx_client import Completion
from src.core.classifier import Domain
from src.core.decoding import FinishReason
from src.core.router import DynamicRouter

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

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0


class SpeculativeEngine:
    def __init__(self, client: LLMClient, router: DynamicRouter):
        self.client = client
        self.router = router

    async def generate(self, payload: dict[str, Any]) -> GenerationResult:
        route = self.router.route(payload)
        start = time.perf_counter()
        completion = await self.client.generate(
            payload.get("messages", []),
            draft_model=route.draft_model,
            num_draft_tokens=route.k_drafts,
            max_tokens=int(payload.get("max_tokens", 256)),
            temperature=float(payload.get("temperature", 0.7)),
        )
        elapsed = time.perf_counter() - start

        if route.draft_model:
            self.router.record_execution_feedback(
                route.domain, completion.accepted_tokens, completion.drafted_tokens
            )
        result = GenerationResult(
            text=completion.text,
            finish_reason=completion.finish_reason,
            domain=route.domain,
            draft_model=route.draft_model,
            accepted_tokens=completion.accepted_tokens,
            drafted_tokens=completion.drafted_tokens,
            cycles=completion.cycles,
        )
        logger.info(
            "domain=%s draft=%s k=%d tokens=%d accepted=%d/%d (%.0f%%) %.2fs (%.1f tok/s)",
            route.domain.value,
            route.draft_model or "none",
            route.k_drafts,
            completion.completion_tokens,
            result.accepted_tokens,
            result.drafted_tokens,
            result.acceptance_rate * 100,
            elapsed,
            completion.completion_tokens / elapsed if elapsed else 0.0,
        )
        return result
