#!/usr/bin/env bash
# Re-run every experiment of the WCNC paper and regenerate all figures.
# Usage: bash scripts/run_all.sh [PROJECT_DIR] [COLORAN_DIR]
set -euo pipefail
cd "$(dirname "$0")"
PROJ=${1:-../../network-slicing-PoC}
COL=${2:-../../coloran_data}
# 0. real-stack traces -> PoC TelemetryRecord format (no change to the PoC)
bash coloran_fetch.sh "$COL" exp1
python coloran_adapter.py --root "$COL/coloran/rome_static_medium"
# E1 grounding / persistence on ColO-RAN
python exp_coloran_grounding.py --project "$PROJ"
# E2 trace-driven closed loop (governed vs ungoverned, fault sweep)
python exp_closed_loop.py --project "$PROJ" --episodes 20
# E3 fault-injection benchmark + LLM-review invariants on ColO-RAN contexts
python exp_governance.py --project "$PROJ" --dataset ../data/coloran_prepared_dataset.csv --outdir coloran
for i in 1 2 3; do  # repeated runs: latency range
  python exp_governance.py --project "$PROJ" --dataset ../data/coloran_prepared_dataset.csv --outdir coloran_rep$i
done
# Earlier pilot (synthetic data): replay of the 4 logged LLM tool calls
python exp_replay_pilot.py --project "$PROJ"
# E5 working-memory scaling on the PoC dataset records (needs redis-server, ~30 min)
python exp_memory_scaling.py --project "$PROJ"
# Optional: E3 on the corrected PoC (v2), reported in Sec. VI
if [ -n "${PROJ_V2:-}" ]; then
  python exp_governance.py --project "$PROJ_V2" --dataset ../data/coloran_prepared_dataset.csv --outdir coloran_v2
fi
# E4: live-LLM runs (mongosh exports results/run-*.txt of the PoC) -> audit + replay
if [ -n "${LIVE_RESULTS:-}" ]; then
  python -I parse_llm_runs.py "$LIVE_RESULTS" ../data/llm_runs
  [ -n "${PROJ_UPLOADED:-}" ] && python replay_live_calls.py --project "$PROJ_UPLOADED" --tag uploaded
  [ -n "${PROJ_V3:-}" ] && python replay_live_calls.py --project "$PROJ_V3" --tag v3 && \
    python exp_governance.py --project "$PROJ_V3" --dataset "$(pwd)/../data/coloran_prepared_dataset.csv" --outdir coloran_v3
fi
if [ -n "${LIVE_RESULTS_B:-}" ] && [ -n "${PROJ_V3_2:-}" ]; then   # Nemotron runs (copy as run-23.txt, run-24.txt)
  python -I parse_llm_runs.py "$LIVE_RESULTS_B" ../data/llm_runs_nemotron
  python replay_live_calls.py --project "$PROJ_V3_2" --runs ../data/llm_runs_nemotron --tag v3_2
  python exp_governance.py --project "$PROJ_V3_2" --dataset "$(pwd)/../data/coloran_prepared_dataset.csv" --outdir coloran_v3_2
  python exp_live_action_check.py
fi
python plot_all.py
python paper_numbers.py
