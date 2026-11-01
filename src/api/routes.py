import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    DomainStatus,
    MetricsResponse,
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

    result = await engine.generate(body.model_dump(exclude_none=True))

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=engine.router.target_model,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        speculative=SpeculativeInfo(
            domain=result.domain,
            draft_model=result.draft_model,
            k=result.metrics.k,
            accepted_tokens=result.accepted_tokens,
            drafted_tokens=result.drafted_tokens,
            acceptance_rate=result.acceptance_rate,
            cycles=result.cycles,
            tokens_per_second=result.metrics.tokens_per_second,
            speedup=result.metrics.speedup,
            dispatch_latency_ms=result.metrics.dispatch_latency_ms,
        ),
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(
    engine: EngineDep, limit: Annotated[int, Query(ge=1, le=1000)] = 200
) -> MetricsResponse:
    domains = [
        DomainStatus(
            domain=domain,
            draft_model=engine.router.draft_registry[domain],
            k=stats.current_k,
            speculation_enabled=stats.speculation_enabled,
            acceptance_rate=stats.acceptance_rate if stats.history else None,
            samples=len(stats.history),
        )
        for domain, stats in engine.router.stats.items()
    ]
    return MetricsResponse(
        target_model=engine.router.target_model,
        baseline_tokens_per_second=engine.metrics.baseline_tps,
        domains=domains,
        requests=engine.metrics.recent(limit),
    )
