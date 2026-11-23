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
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Allow `python scripts/run_loop.py` from the project root (no PYTHONPATH needed).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.globals import set_debug
from mcp import Client

import ric_tools.knowledge
from a2a_agents import settings as a2a_settings
from a2a_agents.cards import MONITORING_SKILL
from a2a_agents.client import A2ACallError, A2AJsonClient
from a2a_agents.monitoring import RemoteMonitoring
from a2a_agents.ric import build_ric_agent
from agents.monitoring_agent import MonitoringAgent
from agents.ran_trust_governance import NON_RT_MIN_PERSISTENCE, RANTrustGovernanceAgent
from agents.record_runner import SUMMARY_KEY, process_record
from agents.record_workflow import RecordWorkflow
from middleware.ran_authorisation import MAX_PROTECTED_CALLS
from ran_emulator.client import EmulatorClient
from ric_tools.client import role_tools
from ric_tools.registry import AgentRole
from scripts.evaluate_run import evaluate
from src.core.llm import PROVIDER_GUARD, create_llm
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
    # "inprocess": every agent in this process. "a2a": every agent is its own
    # service (python -m a2a_agents ...) and this runner only drives the RAN clock.
    agents: Literal["inprocess", "a2a"] = "inprocess"
    monitoring_url: str = field(default_factory=a2a_settings.monitoring_url)


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
        "agents": config.agents,
        **({"monitoring_url": config.monitoring_url} if config.agents == "a2a" else {}),
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
    monitoring: RemoteMonitoring | None = None,
) -> None:
    """Run `config.steps` emulator steps through the agents.

    `emulator`, `mcp_clients` and `monitoring` are for tests (in-process
    emulator, MCP servers and Monitoring service); by default the runner
    connects to the emulator at config.emulator_url, to the MCP servers through
    LiteLLM (in process) or to the Monitoring agent at config.monitoring_url (a2a)."""
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

            if config.agents == "a2a":
                observe = await _remote_observer(stack, config, summary, monitoring)
            else:
                observe = await _local_observer(stack, config, summary, memory,
                                                mongo_repository, mcp_clients)

            mongo_repository.start_run(run_id=config.run_id,
                                       config=_run_config(config, emulator_state),
                                       summary=summary)
            started = True
            logger.info("Run %s: %d steps of scenario %s (agents: %s)", config.run_id,
                        config.steps, emulator_state.get("scenario"), config.agents)
            PROVIDER_GUARD.reset()

            for step in range(config.steps):
                records = await emulator.step_records()
                summary["steps"] += 1
                summary["rows_selected"] += len(records)
                mongo_repository.count(run_id=config.run_id,
                                       counters={"steps": 1, "rows_selected": len(records)})
                stopped = await observe(step, records)
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
        config.run_id, summary["steps"], summary["rows_processed"],
        summary["rows_without_violation"], summary["rows_failed"],
    )


Observer = Callable[[int, list[TelemetryRecord]], Awaitable[str | None]]


async def _local_observer(
    stack: AsyncExitStack,
    config: RunConfig,
    summary: dict[str, Any],
    memory: RedisWorkingMemory,
    mongo_repository: MongoRepository,
    mcp_clients: dict[AgentRole, Client] | None,
) -> Observer:
    """All agents in this process; the RIC agents' tools from their MCP servers."""
    # The sessions stay open for the whole run.
    tools_of = {
        role: await stack.enter_async_context(
            role_tools(role, client=(mcp_clients or {}).get(role)))
        for role in ("near_rt_ric", "non_rt_ric")
    }
    governance_agent = RANTrustGovernanceAgent()
    ric_agents = {
        role: build_ric_agent(role, governance=governance_agent, memory=memory,
                              mongo_repository=mongo_repository, run_id=config.run_id,
                              action_tools=tools_of[role])
        for role in ("near_rt_ric", "non_rt_ric")
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

    async def observe(step: int, records: list[TelemetryRecord]) -> str | None:
        for record in records:
            stopped = await _process_record(workflow, mongo_repository, config,
                                            summary, record, step)
            if stopped is not None:
                return stopped
        return None

    return observe


async def _remote_observer(
    stack: AsyncExitStack,
    config: RunConfig,
    summary: dict[str, Any],
    monitoring: RemoteMonitoring | None,
) -> Observer:
    """Every agent is its own service: send each step to the Monitoring agent
    (A2A), which invokes the RIC agents (A2A), which consult governance."""
    if monitoring is None:
        client = A2AJsonClient(config.monitoring_url, token=a2a_settings.token("monitoring"),
                               timeout_s=a2a_settings.monitoring_timeout_s())
        stack.push_async_callback(client.aclose)
        if MONITORING_SKILL not in await client.skill_ids():
            raise RuntimeError(f"the agent at {config.monitoring_url} does not declare "
                               f"the skill {MONITORING_SKILL}")
        monitoring = RemoteMonitoring(client)

    async def observe(step: int, records: list[TelemetryRecord]) -> str | None:
        try:
            response = await monitoring.observe(config.run_id, step, records)
        except A2ACallError as exc:
            # Fail closed: without the Monitoring agent the loop cannot go on.
            logger.error("Step %d: the Monitoring agent is unavailable: %s", step, exc)
            return f"the Monitoring agent is unavailable (step {step}): {exc}"
        for result in response["results"]:
            summary[SUMMARY_KEY[result["status"]]] += 1
        return response.get("stopped")

    return observe


async def _process_record(
    workflow: RecordWorkflow,
    mongo_repository: MongoRepository,
    config: RunConfig,
    summary: dict[str, Any],
    record: TelemetryRecord,
    step: int,
) -> str | None:
    """Run one record through the workflow (in process). Returns why the run
    must stop (every model of the fallback chain is rate-limited), else None."""
    outcome = await process_record(workflow, mongo_repository, config.run_id, record, step)
    summary[SUMMARY_KEY[outcome.status]] += 1
    return outcome.stopped


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the agents in closed loop on the RAN emulator.")
    parser.add_argument("--scenario", required=True,
                        help="Emulator scenario name (ran_emulator/scenarios/) or path.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Emulator steps to run (default {DEFAULT_STEPS}).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Scenario seed (default: the scenario's own).")
    parser.add_argument("--emulator-url", default=DEFAULT_EMULATOR_URL)
    parser.add_argument("--run-id", default=None,
                        help="Optional run id; defaults to a generated UUID.")
    parser.add_argument("--knowledge-file", type=Path, default=DEFAULT_KNOWLEDGE_FILE,
                        help="RAN knowledge file used by the knowledge search tool.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Parent directory of the run's result files (results/<run_id>/).")
    parser.add_argument("--agents", choices=["inprocess", "a2a"], default="inprocess",
                        help="inprocess: all agents in this process. a2a: each agent is its own "
                             "service (start them with scripts/start_stack.sh --a2a).")
    parser.add_argument("--monitoring-url", default=a2a_settings.monitoring_url(),
                        help="Monitoring agent base URL (its card), for --agents a2a; "
                             "MONITORING_URL by default.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose LangChain tracing (set_debug).")
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
        agents=args.agents,
        monitoring_url=args.monitoring_url,
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
