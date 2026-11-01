import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.clients.mlx_client import MLXClient
from src.core.engine import SpeculativeEngine
from src.core.router import DynamicRouter, RoutingSettings

load_dotenv()

# Third-party libraries stay at WARNING; only this app logs at LOG_LEVEL.
logging.basicConfig(format="%(levelname)s:     %(name)s - %(message)s")
logging.getLogger("src").setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


class _SkipMetricsPolling(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /metrics" not in record.getMessage()


# The dashboard polls /metrics every 2s; keep those out of the access log.
logging.getLogger("uvicorn.access").addFilter(_SkipMetricsPolling())

DEFAULT_ROUTING_CONFIG = "config/routing.yaml"
DASHBOARD_DIST = Path("dashboard/dist")


def create_app(engine: SpeculativeEngine | None = None) -> FastAPI:
    """Pass `engine` to bypass model loading (used by tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client: MLXClient | None = None
        if engine is None:
            settings = RoutingSettings.from_file(
                os.getenv("ROUTING_CONFIG", DEFAULT_ROUTING_CONFIG)
            )
            client = MLXClient(
                settings.target_model,
                (d.draft_model for d in settings.domains.values()),
            )
            await client.load()
            app.state.engine = SpeculativeEngine(client, DynamicRouter.from_settings(settings))
            await app.state.engine.calibrate()
        else:
            app.state.engine = engine
        try:
            yield
        finally:
            if client is not None:
                client.close()

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

    if DASHBOARD_DIST.is_dir():
        app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIST, html=True), name="dashboard")

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse("/dashboard/")

    return app


app = create_app()
