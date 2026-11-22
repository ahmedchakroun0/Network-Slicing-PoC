#!/usr/bin/env bash
# Start the closed-loop stack, each service in the background, in order:
#   RAN emulator (:8100) -> MCP servers near_rt (:8201) / non_rt (:8202) -> LiteLLM (:4000)
# Each one is checked before the next starts. Logs go to logs/<service>.log,
# and Ctrl+C stops everything.
#
#   ./scripts/start_stack.sh [scenario]        # default: embb_prb_starvation
# then, in another terminal:
#   poetry run python scripts/run_loop.py --scenario embb_prb_starvation --steps 10
#
# run_loop.py resets the emulator to its own --scenario: the scenario here is
# only the emulator's initial state.

set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SCENARIO="${1:-embb_prb_starvation}"
mkdir -p logs

if [[ ! -f .env ]]; then
    echo "[ERROR] .env file not found (copy .env.example)"
    exit 1
fi
set -a
source .env
set +a

pids=()
cleanup() {
    echo "[INFO] Stopping the stack"
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# wait_for NAME URL: until the URL answers (any HTTP status), at most 60 s.
wait_for() {
    for _ in $(seq 120); do
        if curl -s -o /dev/null "$2"; then
            echo "[INFO] $1 is up ($2)"
            return 0
        fi
        sleep 0.5
    done
    echo "[ERROR] $1 did not start; see logs/$1.log"
    exit 1
}

start() {
    local name="$1"; shift
    "$@" >"logs/$name.log" 2>&1 &
    pids+=("$!")
}

start emulator poetry run python -m ran_emulator --scenario "$SCENARIO"
wait_for emulator http://127.0.0.1:8100/state

start mcp_near_rt poetry run python -m ric_tools --role near_rt --port 8201
start mcp_non_rt poetry run python -m ric_tools --role non_rt --port 8202
wait_for mcp_near_rt http://127.0.0.1:8201/mcp
wait_for mcp_non_rt http://127.0.0.1:8202/mcp

start litellm poetry run litellm --config litellm_config.yaml --port 4000
wait_for litellm http://127.0.0.1:4000/health/liveliness

echo "[INFO] Stack ready. MCP through LiteLLM: ${LITELLM_API_BASE:-http://localhost:4000}/near_rt/mcp and /non_rt/mcp"
echo "[INFO] Ctrl+C to stop."
wait
