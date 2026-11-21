#!/usr/bin/env bash

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[INFO] Loading environment variables from .env"

if [[ ! -f .env ]]; then
    echo "[ERROR] .env file not found"
    exit 1
fi

set -a
source .env
set +a

if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
    echo "[ERROR] LITELLM_MASTER_KEY is not set"
    exit 1
fi

echo "[INFO] Starting LiteLLM proxy..."
echo "[INFO] Config: $(pwd)/litellm_config.yaml"
echo "[INFO] Port: 4000"
echo "--------------------------------------------------"

# Replace the shell process with LiteLLM.
# All original LiteLLM stdout/stderr logs appear directly in the terminal.
exec poetry run litellm \
    --config litellm_config.yaml \
    --port 4000

Run:

chmod +x scripts/start_litellm.sh
./scripts/start_litellm.sh