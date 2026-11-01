import asyncio
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import mlx.core as mx
from huggingface_hub.utils import disable_progress_bars
from mlx_lm import load
from mlx_lm.generate import wired_limit

from src.core.decoding import FinishReason, speculative_decode, target_decode

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: FinishReason
    completion_tokens: int
    accepted_tokens: int
    drafted_tokens: int
    cycles: int
    tokens_per_second: float
    dispatch_latency_ms: float


class MLXClient:
    """Target and draft models loaded once, in-process, on Apple Silicon (MLX).

    All MLX work runs on a single worker thread: requests are serialised on the
    GPU and the event loop never blocks on inference.
    """

    def __init__(self, target_model: str, draft_models: Iterable[str]):
        self.target_model = target_model
        self._draft_names = sorted(set(draft_models))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._target: Any = None
        self._tokenizer: Any = None
        self._drafts: dict[str, Any] = {}

    async def load(self) -> None:
        await self._run(self._load)

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        draft_model: str | None,
        num_draft_tokens: int,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        return await self._run(
            self._generate, messages, draft_model, num_draft_tokens, max_tokens, temperature
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def _load(self) -> None:
        disable_progress_bars()
        for name in [self.target_model, *self._draft_names]:
            logger.info("Loading %s (downloaded on first use)", name)
            model, tokenizer = load(name)[:2]
            if name == self.target_model:
                self._target, self._tokenizer = model, tokenizer
            else:
                self._drafts[name] = model
        logger.info("Models ready, %.1f GB in memory", mx.get_active_memory() / 1e9)

    def _generate(
        self,
        messages: list[dict[str, str]],
        draft_model: str | None,
        num_draft_tokens: int,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        text = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt = self._tokenizer.encode(text, add_special_tokens=False)
        eos_ids = set(self._tokenizer.eos_token_ids)
        try:
            with wired_limit(self._target):
                if draft_model is None or num_draft_tokens <= 0:
                    out = target_decode(
                        self._target,
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        eos_ids=eos_ids,
                    )
                else:
                    out = speculative_decode(
                        self._target,
                        self._drafts[draft_model],
                        prompt,
                        num_draft_tokens=num_draft_tokens,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        eos_ids=eos_ids,
                    )
        finally:
            # Return MLX's buffer cache to the OS between requests.
            mx.clear_cache()
        return Completion(
            text=self._tokenizer.decode(out.tokens),
            finish_reason=out.finish_reason,
            completion_tokens=len(out.tokens),
            accepted_tokens=out.accepted_tokens,
            drafted_tokens=out.drafted_tokens,
            cycles=out.cycles,
            tokens_per_second=out.tokens_per_second,
            dispatch_latency_ms=out.dispatch_latency_ms,
        )
