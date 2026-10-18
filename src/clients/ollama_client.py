from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float
    # Top-N alternatives at this position; always includes the emitted token.
    top_logprobs: dict[str, float]


@dataclass(frozen=True)
class Completion:
    text: str
    tokens: list[TokenLogprob]
    done_reason: str | None


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        keep_alive: str = "10m",
        http: httpx.AsyncClient | None = None,
    ):
        self._http = http or httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._keep_alive = keep_alive

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
        """Raw-mode completion: `prompt` must already contain the chat template."""
        options: dict[str, Any] = {
            "num_predict": num_predict,
            "temperature": temperature,
            # Disable truncation so the sampled token follows softmax(logits / T).
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
        }
        if seed is not None:
            options["seed"] = seed
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": options,
        }
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs

        resp = await self._http.post("/api/generate", json=body)
        resp.raise_for_status()
        data = resp.json()

        tokens = [
            TokenLogprob(
                token=entry["token"],
                logprob=entry["logprob"],
                top_logprobs={
                    **{t["token"]: t["logprob"] for t in entry.get("top_logprobs") or []},
                    entry["token"]: entry["logprob"],
                },
            )
            for entry in data.get("logprobs") or []
        ]
        return Completion(
            text=data.get("response", ""),
            tokens=tokens,
            done_reason=data.get("done_reason"),
        )

    async def aclose(self) -> None:
        await self._http.aclose()
