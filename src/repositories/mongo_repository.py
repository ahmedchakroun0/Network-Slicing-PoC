"""MongoDB results store: what happened to every telemetry record of a run.

Collections (shapes in src/schema/mongo_run_document.py):
  runs     {_id: run_id, created_at, finished_at, status, config, summary}
  records  {_id: "<run_id>:<record_id>", run_id, seq, status, telemetry, sla,
            persistence, monitoring, agents: {<agent>: {status, ..., actions: [...]}}, error}

Every agent step is one small targeted update of one record document
($setOnInsert / $set / $push); documents are never re-written whole, and
nothing is cached in memory, so the cost of a write does not grow with the run.

  record received   insert_record   $setOnInsert telemetry, sla, persistence
  decision          set_decision    $set monitoring, status="decided"
  agent starts      start_agent     $set agents.<agent>
  each tool call    push_action     $push agents.<agent>.actions (with the governance review)
  knowledge lookup  push_knowledge  $push agents.<agent>.knowledge
  agent ends        finish_agent    $set agents.<agent>.status / finished_at / selected_action
  action verified   set_action_outcome  $set agents.<agent>.actions.<i>.outcome (on the
                                    record that took the action, from a later record)
  record ends       finish_record   $set status="done"; runs: $inc summary counters
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, MongoClient

from src.schema.mongo_run_document import (
    ActionEntry,
    AgentMessage,
    AgentRun,
    KnowledgeEntry,
    RecordDocument,
    RunDocument,
)

DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_MONGO_DB = "network_slicing_poc"


def _now() -> datetime:
    return datetime.now(UTC)


def record_key(run_id: str, record_id: str) -> str:
    return f"{run_id}:{record_id}"


class MongoRepository:
    def __init__(self, mongo_uri: str | None = None, database_name: str | None = None) -> None:
        client = MongoClient(mongo_uri or os.environ.get("MONGO_URI", DEFAULT_MONGO_URI))
        db = client[database_name or os.environ.get("MONGO_DB", DEFAULT_MONGO_DB)]
        self.runs = db["runs"]
        self.records = db["records"]
        self.ensure_indexes()

    def ensure_indexes(self) -> None:
        self.records.create_index([("run_id", ASCENDING), ("seq", ASCENDING)])
        self.records.create_index([("run_id", ASCENDING), ("telemetry.cell_id", ASCENDING),
                                   ("telemetry.slice_id", ASCENDING)])
        self.records.create_index([("run_id", ASCENDING), ("sla.violations", ASCENDING)])

    def _update_record(self, run_id: str, record_id: str, update: dict[str, Any]) -> None:
        self.records.update_one({"_id": record_key(run_id, record_id)}, update)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def start_run(self, *, run_id: str, config: dict[str, Any], summary: dict[str, Any]) -> None:
        doc = RunDocument(_id=run_id, created_at=_now(), config=config, summary=summary)
        fields = doc.model_dump(by_alias=True, exclude={"id"})
        self.runs.update_one({"_id": run_id}, {"$setOnInsert": fields}, upsert=True)

    def count(self, *, run_id: str, counters: dict[str, int]) -> None:
        """Increment run summary counters, e.g. {"rows_processed": 1}."""
        if counters:
            self.runs.update_one({"_id": run_id},
                                 {"$inc": {f"summary.{k}": v for k, v in counters.items()}})

    def finish_run(self, *, run_id: str, stopped: str | None = None) -> None:
        """Close the run; `stopped` is why it ended before its last row."""
        update: dict[str, Any] = {"status": "stopped" if stopped else "finished",
                                  "finished_at": _now()}
        if stopped:
            update["stop_reason"] = stopped
        self.runs.update_one({"_id": run_id}, {"$set": update})

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def insert_record(
        self,
        *,
        run_id: str,
        record_id: str,
        seq: int,
        telemetry: dict[str, Any],
        sla: dict[str, Any],
        persistence: dict[str, int],
    ) -> None:
        doc = RecordDocument(_id=record_key(run_id, record_id), run_id=run_id, seq=seq,
                             status="received", telemetry=telemetry, sla=sla,
                             persistence=persistence)
        fields = doc.model_dump(by_alias=True, exclude={"id", "monitoring", "agents", "error"})
        self.records.update_one({"_id": doc.id}, {"$setOnInsert": fields}, upsert=True)

    def set_decision(self, *, run_id: str, record_id: str, decision: dict[str, Any]) -> None:
        self._update_record(run_id, record_id, {"$set": {
            "monitoring": {**decision, "decided_at": _now()}, "status": "decided"}})

    def finish_record(self, *, run_id: str, record_id: str) -> None:
        self._update_record(run_id, record_id, {"$set": {"status": "done"}})

    def set_llm_usage(self, *, run_id: str, record_id: str, usage: dict[str, Any]) -> None:
        """Model calls and tokens of the record (per caller), and the models
        that answered them (after gateway fallbacks); added to the run totals."""
        served = usage.get("models") or {}
        llm = {key: value for key, value in usage.items() if key != "models"}
        self._update_record(run_id, record_id, {"$set": {
            "llm": llm,
            "models": [{"model": model, "calls": calls} for model, calls in sorted(served.items())],
        }})
        self.count(run_id=run_id, counters={
            f"llm.{key}": llm.get(key, 0)
            for key in ("calls", "failed_calls", "input_tokens", "output_tokens",
                        "reasoning_tokens")})

    def fail_record(
        self, *, run_id: str, record_id: str, seq: int, error: dict[str, Any],
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        """Mark a record failed; creates it when it failed before insert_record."""
        self.records.update_one(
            {"_id": record_key(run_id, record_id)},
            {"$set": {"status": "failed", "error": error},
             "$setOnInsert": {"run_id": run_id, "seq": seq, "telemetry": telemetry or {},
                              "sla": {}, "persistence": {}}},
            upsert=True,
        )

    # ------------------------------------------------------------------
    # RIC agents
    # ------------------------------------------------------------------

    def start_agent(self, *, run_id: str, record_id: str, agent: str) -> None:
        run = AgentRun(status="running", started_at=_now())
        self._update_record(run_id, record_id, {"$set": {f"agents.{agent}": run.model_dump()}})

    def push_action(self, *, run_id: str, record_id: str, agent: str, entry: ActionEntry) -> None:
        self._update_record(run_id, record_id,
                            {"$push": {f"agents.{agent}.actions": entry.model_dump()}})

    def push_knowledge(
        self, *, run_id: str, record_id: str, agent: str, entry: KnowledgeEntry
    ) -> None:
        self._update_record(run_id, record_id,
                            {"$push": {f"agents.{agent}.knowledge": entry.model_dump()}})

    def finish_agent(
        self,
        *,
        run_id: str,
        record_id: str,
        agent: str,
        status: str,
        selected_action: int | None = None,
        final_message: AgentMessage | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        prefix = f"agents.{agent}"
        fields: dict[str, Any] = {f"{prefix}.status": status, f"{prefix}.finished_at": _now(),
                                  f"{prefix}.selected_action": selected_action}
        if final_message is not None:
            fields[f"{prefix}.final_message"] = final_message.model_dump()
        if error is not None:
            fields[f"{prefix}.error"] = error
        self._update_record(run_id, record_id, {"$set": fields})

    def set_action_outcome(
        self, *, run_id: str, record_id: str, agent: str, index: int, outcome: dict[str, Any]
    ) -> None:
        """Verification of an executed action, set on the action itself."""
        self._update_record(run_id, record_id, {"$set": {
            f"agents.{agent}.actions.{index}.outcome": outcome}})
        self.count(run_id=run_id, counters={f"action_outcomes.{outcome['outcome']}": 1})


__all__ = ["MongoRepository", "record_key"]
