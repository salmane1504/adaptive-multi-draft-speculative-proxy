import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.clients.ollama_client import OllamaClient
from src.core.engine import SpeculativeEngine
from src.core.router import DynamicRouter, RoutingSettings

load_dotenv()

DEFAULT_ROUTING_CONFIG = "config/routing.yaml"


def create_app(engine: SpeculativeEngine | None = None) -> FastAPI:
    """Pass `engine` to bypass Ollama wiring (used by tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client: OllamaClient | None = None
        if engine is None:
            settings = RoutingSettings.from_file(
                os.getenv("ROUTING_CONFIG", DEFAULT_ROUTING_CONFIG)
            )
            client = OllamaClient(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            app.state.engine = SpeculativeEngine(client, DynamicRouter.from_settings(settings))
        else:
            app.state.engine = engine
        try:
            yield
        finally:
            if client is not None:
                await client.aclose()

    app = FastAPI(title="Adaptive Multi-Draft Speculative Proxy", lifespan=lifespan)

    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def add_process_time(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    app.include_router(router)
    return app


app = create_app()
