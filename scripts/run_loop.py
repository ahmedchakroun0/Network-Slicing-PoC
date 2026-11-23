"""Run the agentic RAN system in closed loop on the RAN emulator.

Each emulator step (lockstep: the clock waits for the agents) emits one
TelemetryRecord per cell and slice. Every record goes through the record
workflow graph (agents/record_workflow.py): SLA analysis, verification of the
slice's earlier actions, Monitoring Agent, RIC agents, governance. An approved
action is executed through the role's MCP server (via the LiteLLM gateway) on
the emulator and applies from the next step, so the next records show its
effect and the verification step records whether it fixed the violation.

Results go to MongoDB, the agents' working memory stays in Redis, and the
run's result files are written to results/<run_id>/ (scripts/evaluate_run.py).

Start the stack first (scripts/start_stack.sh): emulator, MCP servers, LiteLLM.

    poetry run python scripts/run_loop.py --scenario embb_prb_starvation --steps 10
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow `python scripts/run_loop.py` from the project root (no PYTHONPATH needed).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.globals import set_debug
from mcp import Client

import ric_tools.knowledge
from agents.monitoring_agent import MonitoringAgent
from agents.near_rt_ric_agent import NearRTRICAgent
from agents.non_rt_ric_agent import NonRTRICAgent
from agents.ran_trust_governance import NON_RT_MIN_PERSISTENCE, RANTrustGovernanceAgent
from agents.record_workflow import RecordWorkflow
from middleware.ran_authorisation import MAX_PROTECTED_CALLS
from ran_emulator.client import EmulatorClient
from ric_tools.client import role_tools
from ric_tools.registry import AgentRole
from scripts.evaluate_run import evaluate
from src.core.llm import PROVIDER_GUARD, ModelProviderStopped, create_llm
from src.memory.working_memory import RedisWorkingMemory
from src.repositories.mongo_repository import MongoRepository
from src.schema.monitoring import TelemetryRecord
from src.schema.sla import SLA_PROFILES

logger = logging.getLogger("run_loop")

DEFAULT_KNOWLEDGE_FILE = ric_tools.knowledge.KNOWLEDGE_FILE
DEFAULT_EMULATOR_URL = "http://127.0.0.1:8100"
DEFAULT_STEPS = 30


@dataclass(slots=True)
class RunConfig:
    run_id: str
    scenario: str
    steps: int = DEFAULT_STEPS
    seed: int | None = None
    emulator_url: str = DEFAULT_EMULATOR_URL
    knowledge_file: Path = DEFAULT_KNOWLEDGE_FILE
    results_dir: Path | None = None


def _use_knowledge_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"RAN knowledge file not found: {path}")
    ric_tools.knowledge.KNOWLEDGE_FILE = path


def _run_config(config: RunConfig, emulator_state: dict[str, Any]) -> dict[str, Any]:
    """What the run was configured with, stored once on the run document."""
    return {
        "source": "emulator",
        "scenario": emulator_state.get("scenario", config.scenario),
        "seed": emulator_state.get("seed", config.seed),
        "steps": config.steps,
        "emulator_url": config.emulator_url,
        "model": os.environ.get("MODEL_NAME"),
        "non_rt_min_persistence": NON_RT_MIN_PERSISTENCE,
        "max_protected_calls": MAX_PROTECTED_CALLS,
        "sla_profiles": {
            slice_id: {kpi: term.model_dump() for kpi, term in profile.items()}
            for slice_id, profile in SLA_PROFILES.items()
        },
    }


async def run_loop(
    config: RunConfig,
    *,
    emulator: EmulatorClient | None = None,
    mcp_clients: dict[AgentRole, Client] | None = None,
) -> None:
    """Run `config.steps` emulator steps through the agents.

    `emulator` and `mcp_clients` are for tests (in-process emulator and MCP
    servers); by default the runner connects to the emulator at
    config.emulator_url and to the MCP servers through LiteLLM."""
    _use_knowledge_file(config.knowledge_file)

    memory = RedisWorkingMemory.from_url()
    await memory.reset()
    mongo_repository = MongoRepository()
    summary: dict[str, Any] = {
        "steps": 0,
        "rows_selected": 0,
        "rows_processed": 0,
        "rows_without_violation": 0,
        "rows_failed": 0,
        "action_required": 0,
        "agent_invocations": {"near_rt_ric": 0, "non_rt_ric": 0},
    }
    stopped: str | None = None
    started = False

    async with AsyncExitStack() as stack:
        try:
            if emulator is None:
                emulator = await stack.enter_async_context(EmulatorClient(config.emulator_url))
            emulator_state = await emulator.reset(config.scenario, config.seed)

            # The action tools of each role, from its MCP server; the sessions
            # stay open for the whole run.
            tools_of = {
                role: await stack.enter_async_context(
                    role_tools(role, client=(mcp_clients or {}).get(role))
                )
                for role in ("near_rt_ric", "non_rt_ric")
            }

            governance_agent = RANTrustGovernanceAgent()
            ric_agents = {
                role: cls(
                    trust_governance_agent=governance_agent,
                    memory=memory,
                    mongo_repository=mongo_repository,
                    run_id=config.run_id,
                    action_tools=tools_of[role],
                )
                for role, cls in (("near_rt_ric", NearRTRICAgent), ("non_rt_ric", NonRTRICAgent))
            }
            monitoring_agent = MonitoringAgent(
                llm=create_llm(),
                memory=memory,
                mongo_repository=mongo_repository,
                run_id=config.run_id,
                sub_agents={role: agent.invoke for role, agent in ric_agents.items()},
            )
            workflow = RecordWorkflow(
                monitoring=monitoring_agent,
                memory=memory,
                mongo_repository=mongo_repository,
                run_id=config.run_id,
            )

            mongo_repository.start_run(
                run_id=config.run_id, config=_run_config(config, emulator_state), summary=summary
            )
            started = True
            logger.info(
                "Run %s: %d steps of scenario %s",
                config.run_id,
                config.steps,
                emulator_state.get("scenario"),
            )
            PROVIDER_GUARD.reset()

            for step in range(config.steps):
                records = await emulator.step_records()
                summary["steps"] += 1
                summary["rows_selected"] += len(records)
                mongo_repository.count(
                    run_id=config.run_id, counters={"steps": 1, "rows_selected": len(records)}
                )
                for record in records:
                    stopped = await _process_record(
                        workflow, mongo_repository, config, summary, record, step
                    )
                    if stopped is not None:
                        break
                if stopped is not None:
                    break
        finally:
            # Always close the run, even on Ctrl+C or a crash mid-run.
            if started:
                mongo_repository.finish_run(run_id=config.run_id, stopped=stopped)
            await memory.close()

    if started and config.results_dir is not None:
        # The run is safe in MongoDB: a failure here loses no result.
        try:
            out, _ = evaluate(mongo_repository, [config.run_id], results_dir=config.results_dir)
            logger.info("Result files written to %s/", out)
        except Exception:
            logger.exception("Could not write the result files of run %s", config.run_id)
    logger.info(
        "Run %s finished: %d steps, %d records processed, %d without violation, %d failed",
        config.run_id,
        summary["steps"],
        summary["rows_processed"],
        summary["rows_without_violation"],
        summary["rows_failed"],
    )


async def _process_record(
    workflow: RecordWorkflow,
    mongo_repository: MongoRepository,
    config: RunConfig,
    summary: dict[str, Any],
    record: TelemetryRecord,
    step: int,
) -> str | None:
    """Run one record through the workflow. Returns why the run must stop
    (every model of the fallback chain is rate-limited), else None."""
    record_key = record.record_id
    try:
        PROVIDER_GUARD.take_usage()
        # seq = emulator step: the verification of earlier actions uses it.
        state = await workflow.run(record, seq=step)
        # The agents and governance absorb model errors (fail closed): a
        # record that hit the stopped provider is failed, not done.
        if PROVIDER_GUARD.stopped is not None:
            raise ModelProviderStopped(PROVIDER_GUARD.stopped)

        if state["outcome"] == "skipped":
            summary["rows_without_violation"] += 1
            return None
        decision = state["decision"]
        summary["rows_processed"] += 1
        mongo_repository.set_llm_usage(
            run_id=config.run_id, record_id=record_key, usage=PROVIDER_GUARD.take_usage()
        )
        logger.info(
            "Step %d (%s): situation=%s action_required=%s agents=%s",
            step,
            record_key,
            decision.situation,
            decision.action_required,
            {
                agent: getattr(result, "status", None)
                for agent, result in state["agent_results"].items()
            },
        )
        return None

    except Exception as exc:
        logger.exception("Step %d (%s) failed", step, record_key)
        summary["rows_failed"] += 1
        mongo_repository.fail_record(
            run_id=config.run_id,
            record_id=record_key,
            seq=step,
            telemetry=record.model_dump(mode="json"),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        mongo_repository.count(run_id=config.run_id, counters={"rows_failed": 1})
        if PROVIDER_GUARD.stopped is not None:
            mongo_repository.set_llm_usage(
                run_id=config.run_id, record_id=record_key, usage=PROVIDER_GUARD.take_usage()
            )
            stopped = (
                f"Every model of the fallback chain is rate-limited "
                f"(step {step}, record {record_key}): {PROVIDER_GUARD.stopped}"
            )
            logger.error("Run %s stopped. %s", config.run_id, stopped)
            return stopped
        return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the agents in closed loop on the RAN emulator."
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Emulator scenario name (ran_emulator/scenarios/) or path.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Emulator steps to run (default {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Scenario seed (default: the scenario's own)."
    )
    parser.add_argument("--emulator-url", default=DEFAULT_EMULATOR_URL)
    parser.add_argument(
        "--run-id", default=None, help="Optional run id; defaults to a generated UUID."
    )
    parser.add_argument(
        "--knowledge-file",
        type=Path,
        default=DEFAULT_KNOWLEDGE_FILE,
        help="RAN knowledge file used by the knowledge search tool.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Parent directory of the run's result files (results/<run_id>/).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose LangChain tracing (set_debug)."
    )
    return parser


def parse_args() -> RunConfig:
    args = build_parser().parse_args()
    # Verbose LangChain tracing is opt-in (it is very large and slows runs).
    set_debug(bool(args.debug))
    return RunConfig(
        run_id=args.run_id or str(uuid.uuid4()),
        scenario=args.scenario,
        steps=args.steps,
        seed=args.seed,
        emulator_url=args.emulator_url,
        knowledge_file=args.knowledge_file,
        results_dir=args.results_dir,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = parse_args()
    print(f"run_id: {config.run_id}")
    asyncio.run(run_loop(config))


if __name__ == "__main__":
    main()
