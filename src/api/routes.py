import time
import uuid
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ModelCard,
    ModelList,
    SpeculativeInfo,
)
from src.core.engine import SpeculativeEngine

router = APIRouter()


def get_engine(request: Request) -> SpeculativeEngine:
    engine: SpeculativeEngine = request.app.state.engine
    return engine


EngineDep = Annotated[SpeculativeEngine, Depends(get_engine)]


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/models", response_model=ModelList)
async def list_models(engine: EngineDep) -> ModelList:
    return ModelList(data=[ModelCard(id=engine.router.target_model)])


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest, engine: EngineDep
) -> ChatCompletionResponse:
    if body.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported")

    try:
        result = await engine.generate(body.model_dump(exclude_none=True))
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"Ollama request failed ({type(exc).__name__})"
        ) from exc

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=engine.router.target_model,
        choices=[Choice(message=ChatMessage(role="assistant", content=result.text))],
        speculative=SpeculativeInfo(
            domain=result.domain,
            draft_model=result.draft_model,
            accepted_tokens=result.accepted_tokens,
            drafted_tokens=result.drafted_tokens,
            acceptance_rate=result.acceptance_rate,
            cycles=result.cycles,
        ),
    )
