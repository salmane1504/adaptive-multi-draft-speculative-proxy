from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.classifier import Domain
from src.metrics.store import RequestMetrics


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    # Accepted for OpenAI compatibility; the configured target model is always used.
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = False
    response_format: dict[str, Any] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class SpeculativeInfo(BaseModel):
    domain: Domain
    draft_model: str | None
    k: int
    accepted_tokens: int
    drafted_tokens: int
    acceptance_rate: float
    cycles: int
    tokens_per_second: float
    speedup: float | None
    dispatch_latency_ms: float


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    speculative: SpeculativeInfo


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "mlx"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class DomainStatus(BaseModel):
    domain: Domain
    draft_model: str
    k: int
    speculation_enabled: bool
    # Mean over the router's sliding window; None until the first speculative request.
    acceptance_rate: float | None
    samples: int


class MetricsResponse(BaseModel):
    target_model: str
    baseline_tokens_per_second: float | None
    domains: list[DomainStatus]
    requests: list[RequestMetrics]
