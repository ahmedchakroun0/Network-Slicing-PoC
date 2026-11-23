#!/usr/bin/env bash
# Start the closed-loop stack, each service in the background, in order:
#   RAN emulator (:8100) -> MCP servers near_rt (:8201) / non_rt (:8202) -> LiteLLM (:4000)
# and with --a2a, every agent as its own service (a2a_agents/):
#   -> governance (:8310) -> Near-RT RIC (:8301) / Non-RT RIC (:8302) -> Monitoring (:8300)
# Each one is checked before the next starts. Logs go to logs/<service>.log,
# and Ctrl+C stops everything.
#
#   ./scripts/start_stack.sh [--a2a] [scenario]     # default: embb_prb_starvation
# then, in another terminal:
#   poetry run python scripts/run_loop.py --scenario embb_prb_starvation --steps 10
#   poetry run python scripts/run_loop.py --scenario embb_prb_starvation --steps 10 --agents a2a
#
# run_loop.py resets the emulator to its own --scenario: the scenario here is
# only the emulator's initial state.

set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

A2A=0
SCENARIO="embb_prb_starvation"
for arg in "$@"; do
    case "$arg" in
        --a2a) A2A=1 ;;
        *) SCENARIO="$arg" ;;
    esac
done
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

if [[ "$A2A" == 1 ]]; then
    # Each agent is its own process; they find each other by URL and Agent Card
    # (a2a_agents/settings.py). Order: what a service calls must be up first.
    start agent_governance poetry run python -m a2a_agents governance
    wait_for agent_governance http://127.0.0.1:8310/health
    start agent_near_rt poetry run python -m a2a_agents near_rt_ric
    start agent_non_rt poetry run python -m a2a_agents non_rt_ric
    wait_for agent_near_rt http://127.0.0.1:8301/health
    wait_for agent_non_rt http://127.0.0.1:8302/health
    start agent_monitoring poetry run python -m a2a_agents monitoring
    wait_for agent_monitoring http://127.0.0.1:8300/health
    echo "[INFO] Agents: monitoring :8300, near_rt_ric :8301, non_rt_ric :8302, governance :8310"
fi

echo "[INFO] Stack ready. MCP through LiteLLM: ${LITELLM_API_BASE:-http://localhost:4000}/near_rt/mcp and /non_rt/mcp"
echo "[INFO] Ctrl+C to stop."
wait
