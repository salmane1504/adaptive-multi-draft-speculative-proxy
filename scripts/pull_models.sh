#!/usr/bin/env bash
set -e

echo "=== Pulling Required Ollama Models ==="

# 1. Check if Ollama is installed and running
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed or not in PATH."
    echo "Please install Ollama from https://ollama.com"
    exit 1
fi

if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama daemon is not running."
    echo "Please start it by running 'ollama serve' in another terminal."
    exit 1
fi

# 2. Define the exact model tags required by the proxy's routing table
# Target Model
TARGET_MODEL="qwen2.5:14b"

# Draft Models
DRAFT_CODE="qwen2.5-coder:0.5b"
DRAFT_PROSE="qwen2.5:0.5b"

MODELS=("$TARGET_MODEL" "$DRAFT_CODE" "$DRAFT_PROSE")

# 3. Pull each model
for MODEL in "${MODELS[@]}"; do
    echo "Pulling $MODEL..."
    ollama pull "$MODEL"
done

echo "=== All models pulled successfully! ==="
echo "You can verify them by running: ollama list"