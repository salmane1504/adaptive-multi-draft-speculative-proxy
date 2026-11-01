# Adaptive Multi-Draft Speculative Proxy

OpenAI-compatible FastAPI gateway that runs speculative decoding in-process on Apple Silicon with [MLX](https://github.com/ml-explore/mlx-lm). Each request is classified (code / JSON / prose), routed to a domain-specific draft model, and verified against a larger target model with rejection sampling: the target scores all `K` drafted tokens in a single forward pass and both KV caches are rewound on rejection. The draft depth `K` adapts to the observed acceptance rate.

Requires an Apple Silicon Mac. The target and all drafts are loaded once at startup (about 10 GB for the default 14B 4-bit target).

## Setup

```bash
uv sync                    # Python deps (or: ./scripts/setup_env.sh)
cp .env.example .env
./scripts/pull_models.sh   # optional: pre-download models (otherwise fetched on first start)
cd dashboard && npm install && npm run build && cd ..   # React dashboard (Node 20+)
```

Target and draft models (Hugging Face repo ids) and initial `K` per domain are set in [config/routing.yaml](config/routing.yaml).

## Run

```bash
uv run python main.py
```

Startup loads the models, warms up each draft, and measures the target-only tok/s baseline used for speedup. Then open:

- Dashboard: http://127.0.0.1:8000/ (served from `dashboard/dist`)
- API docs: http://127.0.0.1:8000/docs

To work on the dashboard with hot reload, keep the server running and in a second terminal run `cd dashboard && npm run dev`, then open http://localhost:5173/dashboard/ (API calls are proxied to port 8000).

## Dashboard

- Per-domain cards: routed draft model, current `K`, windowed acceptance `α`, and whether speculation is on.
- Live charts (polled every 2 s): speedup vs the target-only baseline (1.5×–2.8× band), acceptance ratio per domain with the controller thresholds, and the `K` used per request.
- Chat panel: every answer shows the draft model the router chose, plus domain, `K`, accepted/drafted tokens, tok/s, and speedup. Tick **JSON mode** to route to the JSON domain.
- Recent requests table, including dispatch latency.

## Adaptive K (step controller)

After each speculative request, the domain's acceptance rate `α` (mean over the last 50 requests) drives `K`:

- `α > 0.6` → `K = min(K + 1, 8)`
- `α < 0.35` → `K = max(K - 1, 0)`; `K = 0` disables speculation for that domain

## Metrics

An in-memory ring buffer (last 1000 requests, lost on restart) stores, per request: domain, draft model, `K`, acceptance rate, speculative tok/s, baseline tok/s, speedup, and dispatch latency. Tok/s covers the generation phase only (prompt prefill excluded). Dispatch latency is the mean wall time of one target forward pass. The baseline is updated by every target-only request.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness check |
| GET | `/v1/models` | Lists the configured target model |
| POST | `/v1/chat/completions` | Chat completion (`stream=true` is not supported) |
| GET | `/metrics?limit=200` | Per-domain controller state and recent request metrics |

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain why leaves change color."}],"temperature":0,"max_tokens":64}'
```

The response follows the OpenAI schema plus a `speculative` object with the detected domain, draft model, `K`, accepted/drafted tokens, acceptance rate, tok/s, speedup, and dispatch latency. Each request also logs one line with the same stats.

## Configuration

See [.env.example](.env.example): `HOST`, `PORT`, `ROUTING_CONFIG`, `LOG_LEVEL`, `CORS_ORIGINS`.

## Checks

```bash
make check   # ruff + mypy + pytest (no model download required)
```
