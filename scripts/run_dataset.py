"""Run a telemetry dataset through the agentic RAN system.

The script processes a CSV row by row, converts each record into a
TelemetryRecord, sends it through the Monitoring Agent, and lets the agents
and the repository layer persist everything to Mongo. Redis remains the live
working-memory store for the agents. No result files are written.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

# Allow `python scripts/run_dataset.py` from the project root (no PYTHONPATH needed).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.globals import set_debug
import pandas as pd

from agents.monitoring_agent import MonitoringAgent
from agents.near_rt_ric_agent import NearRTRICAgent
from agents.non_rt_ric_agent import NonRTRICAgent
from agents.ran_trust_governance import RANTrustGovernanceAgent
from src.core.llm import create_llm
from src.memory.working_memory import RedisWorkingMemory
from src.repositories.mongo_repository import MongoRepository
from src.schema.monitoring import TelemetryRecord
from tools import near_rt_ric_tools

logger = logging.getLogger("run_dataset")

DEFAULT_KNOWLEDGE_FILE = Path("data/ran_knowledge.txt")
DEFAULT_DATASET_PATH = Path("data/coloran_prepared_sample.csv")
# Target used when a record has no "Cell ID" column (e.g. the synthetic
# 6G slicing dataset); datasets with cell ids use them as the target.
DEFAULT_TARGET = "cell-01"



@dataclass(slots=True)
class RunConfig:
    dataset_path: Path
    run_id: str
    start_row: int
    max_rows: int | None
    knowledge_file: Path


# --------------------------------------------------------------------------
# CSV value helpers
# --------------------------------------------------------------------------


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _truthy(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "violation", "violated"}:
        return True
    if text in {"false", "0", "no", "n", "ok", "none", ""}:
        return False
    return None


def _normalize_violation_name(name: str) -> str:
    """'Packet Loss Violation' -> 'packet_loss' (matches SLAAnalysis names)."""
    text = name.strip().lower()
    if text.endswith(" violation"):
        text = text[: -len(" violation")]
    return text.replace("-", "_").replace(" ", "_")


def _parse_violations(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [str(item) for item in value]
    else:
        text = str(value).strip()
        if not text:
            return []
        for token in (";", "|"):
            text = text.replace(token, ",")
        text = text.strip("[](){}")
        items = text.split(",")

    names = [_normalize_violation_name(item.strip("'\" ")) for item in items]
    return [name for name in names if name and name != "none"]


# --------------------------------------------------------------------------
# Row -> telemetry record / ground truth
# --------------------------------------------------------------------------


def _record_key(row: pd.Series, row_number: int) -> str:
    """Single source of truth for the record id used everywhere (Mongo key,
    TelemetryRecord.record_id, and therefore the Monitoring Agent)."""
    return _optional_str(row.get("record_id")) or f"row-{row_number}"


def _row_to_telemetry_record(row: pd.Series, record_key: str) -> TelemetryRecord:
    return TelemetryRecord(
        record_id=record_key,
        use_case_type=_optional_str(row.get("Use Case Type")),
        packet_loss_budget=_optional_float(row.get("Packet Loss Budget")),
        latency_budget_ns=_optional_float(row.get("Latency Budget (ns)")),
        jitter_budget_ns=_optional_float(row.get("Jitter Budget (ns)")),
        data_rate_budget_gbps=_optional_float(row.get("Data Rate Budget (Gbps)")),
        required_mobility=_optional_str(row.get("Required Mobility")),
        required_connectivity=_optional_str(row.get("Required Connectivity")),
        slice_available_transfer_rate_gbps=_optional_float(
            row.get("Slice Available Transfer Rate (Gbps)")
        ),
        slice_latency_ns=_optional_float(row.get("Slice Latency (ns)")),
        slice_packet_loss=_optional_float(row.get("Slice Packet Loss")),
        slice_jitter_ns=_optional_float(row.get("Slice Jitter (ns)")),
        slice_type=_optional_str(row.get("Slice Type")),
        slice_handover=_optional_str(row.get("Slice Handover")),
        cell_id=_optional_str(row.get("Cell ID")),
    )


def _ground_truth_payload(row: pd.Series, record_key: str) -> dict[str, Any]:
    return {
        "record_id": record_key,
        "metric_flags": {
            "latency": _truthy(row.get("Latency Violation")),
            "jitter": _truthy(row.get("Jitter Violation")),
            "packet_loss": _truthy(row.get("Packet Loss Violation")),
            "throughput": _truthy(row.get("Throughput Violation")),
        },
        "sla_violations_raw": _optional_str(row.get("SLA Violations")),
        "sla_violations": _parse_violations(row.get("SLA Violations")),
        "sla_evaluation_status": _optional_str(row.get("SLA Evaluation Status")),
    }


def _compare_ground_truth(
    record: TelemetryRecord, row: pd.Series, record_key: str
) -> dict[str, Any]:
    analysis = MonitoringAgent.analyze_sla(record)
    gt = _ground_truth_payload(row, record_key)
    expected_flags = gt["metric_flags"]
    computed_flags = {
        name: name in analysis.violations
        for name in ("latency", "jitter", "packet_loss", "throughput")
    }
    metric_matches = {
        name: (
            expected_flags[name] is None or expected_flags[name] == computed_flags[name]
        )
        for name in computed_flags
    }

    return {
        "computed_analysis": analysis.model_dump(mode="json"),
        "ground_truth": gt,
        "comparison": {
            "metric_matches": metric_matches,
            "sla_violations_match": (
                not gt["sla_violations"]
                or sorted(gt["sla_violations"]) == sorted(analysis.violations)
            ),
            "evaluation_status_match": (
                gt["sla_evaluation_status"] is None
                or gt["sla_evaluation_status"].lower()
                == ("violation" if analysis.violations else "ok")
            ),
        },
    }


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
    # Context the RIC agents and the governance agent depend on. It comes from
    # deterministic code, never from the Monitoring LLM, and it overrides
    # anything the LLM put in `selected_agents[].context`.
    trusted_context: dict[str, Any] = {}


    def with_trusted_context(invoke):
        async def _invoke(invocation):
            context = {**(invocation.context or {}), **trusted_context}
            return await invoke(invocation.model_copy(update={"context": context}))

        return _invoke

    monitoring_agent = MonitoringAgent(
        llm=create_llm(),
        memory=memory,
        mongo_repository=mongo_repository,
        run_id=config.run_id,
        sub_agents={
            "near_rt_ric": with_trusted_context(near_rt_agent.invoke),
            "non_rt_ric": with_trusted_context(non_rt_agent.invoke),
        },
    )

    dataframe = pd.read_csv(config.dataset_path)
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
        "rows_failed": 0,
        "monitoring_action_required": 0,
        "agent_invocations": {"near_rt_ric": 0, "non_rt_ric": 0},
    }
    mongo_repository.ensure_run(run_id=config.run_id, summary=summary)
    logger.info(
        "Run %s: processing %d rows", config.run_id, summary["rows_selected"]
    )

    try:
        for offset, (_, row) in enumerate(selected_rows.iterrows(), start=start_index):
            row_number = offset + 1
            record_key = _record_key(row, row_number)
            telemetry_dump: dict[str, Any] = {}

            try:
                telemetry_record = _row_to_telemetry_record(row, record_key)
                telemetry_dump = telemetry_record.model_dump(mode="json")
                comparison = _compare_ground_truth(telemetry_record, row, record_key)

                analysis = comparison["computed_analysis"]
                target = telemetry_record.cell_id or DEFAULT_TARGET
                slice_id = telemetry_record.slice_type or record_key
                
                trusted_context.clear()
                trusted_context.update(
                    target=target,
                    slice_id=slice_id,
                    sla_analysis=analysis,
                )

                mongo_repository.start_record(
                    run_id=config.run_id,
                    record_document=mongo_repository.new_record_document(
                        record_id=record_key,
                        telemetry_record=telemetry_dump,
                        ground_truth=comparison["ground_truth"],
                        comparison=comparison["comparison"],
                        computed_analysis=comparison["computed_analysis"],
                    ),
                )

                decision = await monitoring_agent.run(
                    telemetry_record
                )

                if decision.action_required:
                    summary["monitoring_action_required"] += 1
                for invocation in decision.selected_agents:
                    summary["agent_invocations"][invocation.agent] += 1
                summary["rows_processed"] += 1

                logger.info(
                    "Row %d (%s): situation=%s action_required=%s agents=%s",
                    row_number,
                    record_key,
                    decision.situation,
                    decision.action_required,
                    [i.agent for i in decision.selected_agents],
                )

            except Exception as exc:
                logger.exception("Row %d (%s) failed", row_number, record_key)
                summary["rows_failed"] += 1
                mongo_repository.start_record(
                    run_id=config.run_id,
                    record_document=mongo_repository.new_record_document(
                        record_id=record_key,
                        telemetry_record=telemetry_dump,
                        error={"type": type(exc).__name__, "message": str(exc)},
                    ),
                )
    finally:
        # Always persist the summary, even on Ctrl+C or a crash mid-run.
        mongo_repository.ensure_run(run_id=config.run_id, summary=summary)
        await memory.close()
        logger.info("Run %s finished: %s", config.run_id, summary)


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
        "--knowledge-file",
        type=Path,
        default=DEFAULT_KNOWLEDGE_FILE,
        help="RAN knowledge file used by the Near-RT knowledge search tool.",
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