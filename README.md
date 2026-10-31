# Adaptive Multi-Draft Speculative Proxy

OpenAI-compatible FastAPI gateway that runs speculative decoding in-process on Apple Silicon with [MLX](https://github.com/ml-explore/mlx-lm). Each request is classified (code / JSON / prose), routed to a domain-specific draft model, and verified against a larger target model with rejection sampling: the target scores all `K` drafted tokens in a single forward pass and both KV caches are rewound on rejection. The draft depth `K` adapts to the observed acceptance rate.

Requires an Apple Silicon Mac. The target and all drafts are loaded once at startup (about 10 GB for the default 14B 4-bit target).

## Setup

```bash
./scripts/setup_env.sh && source .venv/bin/activate   # or: uv sync
cp .env.example .env
./scripts/pull_models.sh   # optional: pre-download models (otherwise fetched on first start)
```

Target and draft models (Hugging Face repo ids) and initial `K` per domain are set in [config/routing.yaml](config/routing.yaml).

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

The response follows the OpenAI schema plus a `speculative` object with the detected domain, draft model, accepted/drafted tokens, and acceptance rate. Each request also logs one line with the same stats and tokens/s.

## Configuration

See [.env.example](.env.example): `HOST`, `PORT`, `ROUTING_CONFIG`, `LOG_LEVEL`, `CORS_ORIGINS`.

## Checks

```bash
make check   # ruff + mypy + pytest (no model download required)
```
