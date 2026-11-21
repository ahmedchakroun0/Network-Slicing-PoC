"""Run a telemetry dataset through the agentic RAN system.

The script processes a CSV row by row, converts each record into a
TelemetryRecord and runs it through the record workflow graph
(agents/record_workflow.py): SLA analysis, Monitoring Agent, RIC agents,
governance. The nodes persist the results to MongoDB; Redis remains the
agents' working memory across records. At the end, the run's result files are
written to results/<run_id>/ (scripts/evaluate_run.py).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow `python scripts/run_dataset.py` from the project root (no PYTHONPATH needed).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from langchain_core.globals import set_debug

from agents.monitoring_agent import MonitoringAgent
from agents.near_rt_ric_agent import NearRTRICAgent
from agents.non_rt_ric_agent import NonRTRICAgent
from agents.ran_trust_governance import NON_RT_MIN_PERSISTENCE, RANTrustGovernanceAgent
from agents.record_workflow import RecordWorkflow
from middleware.ran_authorisation import MAX_PROTECTED_CALLS
from scripts.evaluate_run import evaluate
from src.core.llm import PROVIDER_GUARD, ModelProviderStopped, create_llm
from src.memory.working_memory import RedisWorkingMemory
from src.repositories.mongo_repository import MongoRepository
from src.schema.monitoring import TelemetryRecord
from src.schema.sla import SLA_PROFILES
from tools import near_rt_ric_tools

logger = logging.getLogger("run_dataset")

DEFAULT_KNOWLEDGE_FILE = Path("data/ran_knowledge.txt")
DEFAULT_DATASET_PATH = Path("data/coloran_prepared_sample.csv")
TELEMETRY_FIELDS = frozenset(TelemetryRecord.model_fields)



@dataclass(slots=True)
class RunConfig:
    dataset_path: Path
    run_id: str
    start_row: int
    max_rows: int | None
    knowledge_file: Path
    results_dir: Path | None = None
    # Only these records (dataset order), e.g. the same records over repeated
    # runs for a stability study; start_row / max_rows then do not apply.
    record_ids: list[str] | None = None


# --------------------------------------------------------------------------
# Row -> telemetry record
# --------------------------------------------------------------------------


def _row_to_telemetry_record(row: pd.Series) -> TelemetryRecord:
    """The prepared CSV columns are the TelemetryRecord fields (empty -> None)."""
    return TelemetryRecord.model_validate(
        {k: (None if pd.isna(v) else v) for k, v in row.items() if k in TELEMETRY_FIELDS}
    )


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def _use_knowledge_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"RAN knowledge file not found: {path}")
    near_rt_ric_tools.KNOWLEDGE_FILE = path


async def run_dataset(config: RunConfig) -> None:
    _use_knowledge_file(config.knowledge_file)

    memory = RedisWorkingMemory.from_url()
    await memory.reset()
    mongo_repository = MongoRepository()

    governance_agent = RANTrustGovernanceAgent()
    near_rt_agent = NearRTRICAgent(
        trust_governance_agent=governance_agent,
        memory=memory,
        mongo_repository=mongo_repository,
        run_id=config.run_id,
    )
    non_rt_agent = NonRTRICAgent(
        trust_governance_agent=governance_agent,
        memory=memory,
        mongo_repository=mongo_repository,
        run_id=config.run_id,
    )
    monitoring_agent = MonitoringAgent(
        llm=create_llm(),
        memory=memory,
        mongo_repository=mongo_repository,
        run_id=config.run_id,
        sub_agents={
            "near_rt_ric": near_rt_agent.invoke,
            "non_rt_ric": non_rt_agent.invoke,
        },
    )
    # The workflow of one record; the trusted context of the RIC agents is
    # built from its state by deterministic code.
    workflow = RecordWorkflow(
        monitoring=monitoring_agent,
        memory=memory,
        mongo_repository=mongo_repository,
        run_id=config.run_id,
    )

    dataframe = pd.read_csv(config.dataset_path)
    if config.record_ids:
        selected_rows = dataframe[dataframe["record_id"].isin(config.record_ids)]
        missing = sorted(set(config.record_ids) - set(selected_rows["record_id"]))
        if missing:
            raise ValueError(f"record ids not in {config.dataset_path}: {', '.join(missing)}")
    else:
        start_index = max(config.start_row - 1, 0)
        end_index = (
            len(dataframe)
            if config.max_rows is None
            else min(len(dataframe), start_index + config.max_rows)
        )
        selected_rows = dataframe.iloc[start_index:end_index]

    summary: dict[str, Any] = {
        "rows_selected": len(selected_rows),
        "rows_processed": 0,
        "rows_without_violation": 0,
        "rows_failed": 0,
        "action_required": 0,
        "agent_invocations": {"near_rt_ric": 0, "non_rt_ric": 0},
    }
    mongo_repository.start_run(run_id=config.run_id, config=_run_config(config), summary=summary)
    logger.info("Run %s: processing %d rows", config.run_id, summary["rows_selected"])
    PROVIDER_GUARD.reset()
    stopped: str | None = None

    try:
        for index, row in selected_rows.iterrows():
            row_number = int(index) + 1   # row in the dataset, whatever the selection
            record_key = str(row.get("record_id") or f"row-{row_number}")
            telemetry_dump: dict[str, Any] | None = None

            try:
                telemetry_record = _row_to_telemetry_record(row)
                telemetry_dump = telemetry_record.model_dump(mode="json")
                PROVIDER_GUARD.take_usage()
                state = await workflow.run(telemetry_record, seq=row_number)
                # The agents and governance absorb model errors (fail closed):
                # a record that hit the stopped provider is failed, not done.
                if PROVIDER_GUARD.stopped is not None:
                    raise ModelProviderStopped(PROVIDER_GUARD.stopped)

                if state["outcome"] == "skipped":
                    summary["rows_without_violation"] += 1
                    continue
                decision = state["decision"]
                summary["rows_processed"] += 1
                mongo_repository.set_llm_usage(run_id=config.run_id, record_id=record_key,
                                               usage=PROVIDER_GUARD.take_usage())

                logger.info(
                    "Row %d (%s): situation=%s action_required=%s agents=%s",
                    row_number,
                    record_key,
                    decision.situation,
                    decision.action_required,
                    {agent: getattr(result, "status", None)
                     for agent, result in state["agent_results"].items()},
                )

            except Exception as exc:
                logger.exception("Row %d (%s) failed", row_number, record_key)
                summary["rows_failed"] += 1
                mongo_repository.fail_record(
                    run_id=config.run_id,
                    record_id=record_key,
                    seq=row_number,
                    telemetry=telemetry_dump,
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                mongo_repository.count(run_id=config.run_id, counters={"rows_failed": 1})
                if PROVIDER_GUARD.stopped is not None:
                    mongo_repository.set_llm_usage(run_id=config.run_id, record_id=record_key,
                                                   usage=PROVIDER_GUARD.take_usage())
                    stopped = (f"Every model of the fallback chain is rate-limited "
                               f"(row {row_number}): {PROVIDER_GUARD.stopped}")
                    logger.error("Run %s stopped. %s", config.run_id, stopped)
                    break
    finally:
        # Always close the run, even on Ctrl+C or a crash mid-run.
        mongo_repository.finish_run(run_id=config.run_id, stopped=stopped)
        await memory.close()
        if config.results_dir is not None:
            # The run is safe in MongoDB: a failure here loses no result.
            try:
                out, _ = evaluate(mongo_repository, [config.run_id],
                                  results_dir=config.results_dir)
                logger.info("Result files written to %s/", out)
            except Exception:
                logger.exception("Could not write the result files of run %s", config.run_id)
        logger.info(
            "Run %s finished: %d processed, %d without violation (skipped), %d failed",
            config.run_id, summary["rows_processed"], summary["rows_without_violation"],
            summary["rows_failed"],
        )


def _run_config(config: RunConfig) -> dict[str, Any]:
    """What the run was configured with, stored once on the run document."""
    return {
        "dataset": str(config.dataset_path),
        "start_row": config.start_row,
        "max_rows": config.max_rows,
        "record_ids": config.record_ids,
        "model": os.environ.get("MODEL_NAME"),
        "non_rt_min_persistence": NON_RT_MIN_PERSISTENCE,
        "max_protected_calls": MAX_PROTECTED_CALLS,
        "sla_profiles": {
            slice_id: {kpi: term.model_dump() for kpi, term in profile.items()}
            for slice_id, profile in SLA_PROFILES.items()
        },
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a CSV telemetry dataset through the agent system."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET_PATH,
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id; defaults to a generated UUID.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based starting data row (after the header).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process.",
    )
    parser.add_argument(
        "--record-ids",
        default=None,
        help="Comma-separated record ids to process (instead of --start-row/--max-rows).",
    )
    parser.add_argument(
        "--knowledge-file",
        type=Path,
        default=DEFAULT_KNOWLEDGE_FILE,
        help="RAN knowledge file used by the Near-RT knowledge search tool.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Parent directory of the run's result files (results/<run_id>/).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose LangChain tracing (set_debug).",
    )
    return parser


def parse_args() -> RunConfig:
    args = build_parser().parse_args()
    # Verbose LangChain tracing is opt-in (it is very large and slows runs).
    set_debug(bool(args.debug))
    return RunConfig(
        dataset_path=args.dataset_path,
        run_id=args.run_id or str(uuid.uuid4()),
        start_row=args.start_row,
        max_rows=args.max_rows,
        knowledge_file=args.knowledge_file,
        results_dir=args.results_dir,
        record_ids=[r.strip() for r in args.record_ids.split(",") if r.strip()]
        if args.record_ids else None,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = parse_args()
    print(f"run_id: {config.run_id}")
    asyncio.run(run_dataset(config))


if __name__ == "__main__":
    main()