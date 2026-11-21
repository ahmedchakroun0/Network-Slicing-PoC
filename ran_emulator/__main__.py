"""Run the RAN emulator API.

    poetry run python -m ran_emulator --scenario baseline_bs2
    poetry run python -m ran_emulator --scenario cell_interference --mode realtime --step-period 1
"""

from __future__ import annotations

import argparse
import logging
import time

import uvicorn

from ran_emulator.api import create_app
from ran_emulator.scenarios import SCENARIO_DIR, load_scenario
from ran_emulator.traces import TraceBank


def main() -> None:
    known = ", ".join(sorted(p.stem for p in SCENARIO_DIR.glob("*.json")))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="baseline_bs2", help=f"name or path ({known})")
    p.add_argument("--seed", type=int, default=None, help="override the scenario seed")
    p.add_argument("--dataset", default=None,
                   help="prepared ColO-RAN CSV (default: full dataset, else the sample)")
    p.add_argument("--mode", choices=["lockstep", "realtime"], default="lockstep")
    p.add_argument("--step-period", type=float, default=1.0, help="seconds per step (realtime)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scenario = load_scenario(a.scenario)
    if a.seed is not None:
        scenario = scenario.model_copy(update={"seed": a.seed})
    t0 = time.perf_counter()
    bank = TraceBank.load(a.dataset)
    logging.info("loaded %d ColO-RAN traces (%s) in %.1f s", len(bank), ", ".join(bank.cells),
                 time.perf_counter() - t0)
    app = create_app(bank, scenario, mode=a.mode, step_period_s=a.step_period)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
