from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.classifier import Domain


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
    finish_reason: Literal["stop"] = "stop"


class SpeculativeInfo(BaseModel):
    domain: Domain
    draft_model: str | None
    accepted_tokens: int
    drafted_tokens: int
    acceptance_rate: float
    cycles: int


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
    owned_by: str = "ollama"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
