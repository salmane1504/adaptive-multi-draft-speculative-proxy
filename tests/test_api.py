from fastapi.testclient import TestClient

from src.api.main import create_app
from src.core.engine import SpeculativeEngine
from src.core.router import DynamicRouter, RoutingSettings
from tests.test_engine import PROMPT, TEXT, FakeClient


def make_client() -> TestClient:
    engine = SpeculativeEngine(FakeClient(), DynamicRouter(target_model="target"))
    return TestClient(create_app(engine))


def test_health():
    with make_client() as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_list_models_returns_target_model():
    with make_client() as client:
        assert client.get("/v1/models").json()["data"][0]["id"] == "target"


def test_chat_completion_is_openai_shaped():
    body = {**PROMPT, "temperature": 0.0, "max_tokens": 32}
    with make_client() as client:
        resp = client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "target"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": TEXT}
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["speculative"]["domain"] == "prose"
    assert data["speculative"]["drafted_tokens"] > 0
    assert "x-process-time-ms" in resp.headers


def test_stream_is_rejected():
    with make_client() as client:
        resp = client.post("/v1/chat/completions", json={**PROMPT, "stream": True})
    assert resp.status_code == 400


def test_invalid_request_is_422():
    with make_client() as client:
        assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 422


def test_routing_yaml_loads_into_router():
    settings = RoutingSettings.from_file("config/routing.yaml")
    router = DynamicRouter.from_settings(settings)
    route = router.route(PROMPT)

    assert route.target_model == "mlx-community/Qwen2.5-14B-Instruct-4bit"
    assert route.draft_model == "mlx-community/Qwen2.5-0.5B-Instruct-8bit"
    assert route.k_drafts == 3
