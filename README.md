# Adaptive Multi-Draft Speculative Proxy

OpenAI-compatible FastAPI gateway that runs speculative decoding on top of local Ollama models. Each request is classified (code / JSON / prose), routed to a domain-specific draft model, and verified against a larger target model with rejection sampling. The draft depth `K` adapts to the observed acceptance rate.

## Setup

```bash
./scripts/setup_env.sh && source .venv/bin/activate   # or: uv sync
cp .env.example .env
```

Start Ollama with enough parallelism for the K+1 verification requests, then pull the models:

```bash
OLLAMA_NUM_PARALLEL=9 OLLAMA_MAX_LOADED_MODELS=3 ollama serve
./scripts/pull_models.sh
```

Target and draft models and initial `K` per domain are set in [config/routing.yaml](config/routing.yaml).

## Run

```bash
python main.py
```

Or: `uvicorn src.api.main:app --reload`. Interactive docs are at http://127.0.0.1:8000/docs.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness check |
| GET | `/v1/models` | Lists the configured target model |
| POST | `/v1/chat/completions` | Chat completion (`stream=true` is not supported) |

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain why leaves change color."}],"temperature":0,"max_tokens":64}'
```

The response follows the OpenAI schema plus a `speculative` object with the detected domain, draft model, accepted/drafted tokens, and acceptance rate.

## Configuration

See [.env.example](.env.example): `HOST`, `PORT`, `OLLAMA_BASE_URL`, `ROUTING_CONFIG`, `CORS_ORIGINS`.

## Checks

```bash
make check   # ruff + mypy + pytest (no Ollama required)
```
