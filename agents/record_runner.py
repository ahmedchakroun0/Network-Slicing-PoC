"""Process one telemetry record through the record workflow.

Shared by the in-process runner (scripts/run_loop.py) and the Monitoring
agent's A2A service (a2a_agents/monitoring.py), so both deployments handle
failures, model usage and a stopped provider the same way.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from agents.record_workflow import RecordWorkflow
from src.core.llm import PROVIDER_GUARD, ModelProviderStopped
from src.repositories.mongo_repository import MongoRepository
from src.schema.monitoring import TelemetryRecord

logger = logging.getLogger(__name__)

RecordStatus = Literal["skipped", "processed", "failed"]
# Run summary counter of each status.
SUMMARY_KEY: dict[RecordStatus, str] = {
    "skipped": "rows_without_violation", "processed": "rows_processed", "failed": "rows_failed"}


@dataclass(slots=True)
class RecordOutcome:
    record_id: str
    status: RecordStatus
    agents: dict[str, str | None] = field(default_factory=dict)  # RIC agent -> result status
    stopped: str | None = None   # why the run must stop (every model rate-limited)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


async def process_record(
    workflow: RecordWorkflow,
    mongo_repository: MongoRepository,
    run_id: str,
    record: TelemetryRecord,
    step: int,
) -> RecordOutcome:
    """Run one record through the workflow; record a failure in MongoDB."""
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
            return RecordOutcome(record_key, "skipped")
        decision = state["decision"]
        mongo_repository.set_llm_usage(run_id=run_id, record_id=record_key,
                                       usage=PROVIDER_GUARD.take_usage())
        agents = {agent: getattr(result, "status", None)
                  for agent, result in state["agent_results"].items()}
        logger.info("Step %d (%s): situation=%s action_required=%s agents=%s",
                    step, record_key, decision.situation, decision.action_required, agents)
        return RecordOutcome(record_key, "processed", agents)

    except Exception as exc:
        logger.exception("Step %d (%s) failed", step, record_key)
        mongo_repository.fail_record(
            run_id=run_id, record_id=record_key, seq=step,
            telemetry=record.model_dump(mode="json"),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        mongo_repository.count(run_id=run_id, counters={"rows_failed": 1})
        stopped = None
        if PROVIDER_GUARD.stopped is not None:
            mongo_repository.set_llm_usage(run_id=run_id, record_id=record_key,
                                           usage=PROVIDER_GUARD.take_usage())
            stopped = (f"Every model of the fallback chain is rate-limited "
                       f"(step {step}, record {record_key}): {PROVIDER_GUARD.stopped}")
            logger.error("Run %s stopped. %s", run_id, stopped)
        return RecordOutcome(record_key, "failed", stopped=stopped)
