#!/usr/bin/env bash
set -e

# Pre-download the MLX models listed in config/routing.yaml into the Hugging Face
# cache so the server starts quickly. Requires Apple Silicon.
PYTHON="${PYTHON:-.venv/bin/python}"
CONFIG="${ROUTING_CONFIG:-config/routing.yaml}"

"$PYTHON" - "$CONFIG" <<'EOF'
import sys

import yaml
from huggingface_hub import snapshot_download

cfg = yaml.safe_load(open(sys.argv[1]))
models = [cfg["target_model"], *sorted({d["draft_model"] for d in cfg["domains"].values()})]
for model in models:
    print(f"Pulling {model}...")
    snapshot_download(model)
print("All models are in ~/.cache/huggingface/hub")
EOF