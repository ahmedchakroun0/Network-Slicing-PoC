from __future__ import annotations

"""Mongo repository scaffold for persisted agent-execution runs.

This repository only models the persisted experiment content.
It does not store Redis runtime state, dataset paths, or other transient
execution metadata.

Document shape
--------------
{
  "_id": "<run_id>",
  "run_id": "<run_id>",
  "summary": {...},
  "records": {
    "<record_id>": {
      "telemetry_record": {...},
      "ground_truth": {...},
      "comparison": {...},
      "computed_analysis": {...},
      "monitoring_decision": {...},
      "agent_executions": {
        "near_rt_ric": {
          "invocation": {...},
          "result": {...},
          "middleware_audit": [...]
                },
                "non_rt_ric": {
                    "invocation": {...},
                    "result": {...},
                    "middleware_audit": [...]
        }
      },
      "error": {...}
    }
  }
}
"""

from dataclasses import dataclass, field
from typing import Any

from src.schema.monitoring import AgentInvocation
from src.schema.mongo_run_document import (
    MongoAgentExecutionDocument,
    MongoDatasetRecordDocument,
    MongoMiddlewareAuditDocument,
    MongoRunDocument,
)
from pymongo import MongoClient


@dataclass(slots=True)
class MongoRepository:
    database_name: str = "network_slicing_poc"
    collection_name: str = "dataset_runs"
    mongo_uri: str = "mongodb://localhost:27017"
    _runs: dict[str, MongoRunDocument] = field(default_factory=dict, init=False, repr=False)
    _collection: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        client = MongoClient(self.mongo_uri)
        self._collection = client[self.database_name][self.collection_name]
    def ensure_run(
        self,
        *,
        run_id: str,
        summary: dict[str, Any] | None = None,
    ) -> MongoRunDocument:
        run_document = self._runs.get(run_id)
        if run_document is None:
            run_document = self.new_run_document(run_id=run_id, summary=summary)
            self._runs[run_id] = run_document
        elif summary:
            run_document.summary.update(summary)
        self.save_run(run_document)
        return run_document

    @staticmethod
    def new_run_document(
        *,
        run_id: str,
        summary: dict[str, Any] | None = None,
    ) -> MongoRunDocument:
        return MongoRunDocument(
            _id=run_id,
            run_id=run_id,
            summary=summary or {},
        )

    @staticmethod
    def new_record_document(
        *,
        record_id: str,
        telemetry_record: dict[str, Any],
        ground_truth: dict[str, Any] | None = None,
        comparison: dict[str, Any] | None = None,
        computed_analysis: dict[str, Any] | None = None,
        monitoring_decision: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> MongoDatasetRecordDocument:
        return MongoDatasetRecordDocument(
            record_id=record_id,
            telemetry_record=telemetry_record,
            ground_truth=ground_truth or {},
            comparison=comparison or {},
            computed_analysis=computed_analysis or {},
            monitoring_decision=monitoring_decision,
            error=error,
        )

    @staticmethod
    def new_agent_execution_document(
        *,
        agent: str,
        invocation: AgentInvocation,
        result: dict[str, Any] | None = None,
        middleware_audit: list[dict[str, Any]] | None = None,
    ) -> MongoAgentExecutionDocument:
        audit_documents: list[MongoMiddlewareAuditDocument] = []
        for entry in middleware_audit or []:
            audit_documents.append(
                MongoMiddlewareAuditDocument(
                    tool_call_id=str(entry.get("tool_call_id") or ""),
                    proposal=entry.get("proposal"),
                    governance=entry["governance"],
                    execution=entry["execution"],
                )
            )

        return MongoAgentExecutionDocument(
            agent=agent,  # type: ignore[arg-type]
            invocation=invocation,
            result=result,
            middleware_audit=audit_documents,
        )

    @staticmethod
    def attach_record(
        run_document: MongoRunDocument,
        record_document: MongoDatasetRecordDocument,
    ) -> MongoRunDocument:
        run_document.upsert_record(record_document)
        return run_document

    def start_record(
        self,
        *,
        run_id: str,
        record_document: MongoDatasetRecordDocument,
        summary: dict[str, Any] | None = None,
    ) -> MongoDatasetRecordDocument:
        self.ensure_run(run_id=run_id, summary=summary)
        run_document = self._runs[run_id]
        existing = run_document.get_record(record_document.record_id)
        if existing is None:
            run_document.upsert_record(record_document)
            self.save_run(run_document)
            return record_document

        existing.telemetry_record = (
            record_document.telemetry_record or existing.telemetry_record
        )
        if record_document.ground_truth:
            existing.ground_truth = record_document.ground_truth
        if record_document.comparison:
            existing.comparison = record_document.comparison
        if record_document.computed_analysis:
            existing.computed_analysis = record_document.computed_analysis
        if record_document.monitoring_decision is not None:
            existing.monitoring_decision = record_document.monitoring_decision
        if record_document.error is not None:
            existing.error = record_document.error
        self.save_run(run_document)
        return existing

    @staticmethod
    def attach_agent_execution(
        record_document: MongoDatasetRecordDocument,
        execution_document: MongoAgentExecutionDocument,
    ) -> MongoDatasetRecordDocument:
        record_document.agent_executions[execution_document.agent] = execution_document
        return record_document

    def set_monitoring_decision(
        self,
        *,
        run_id: str,
        record_id: str,
        monitoring_decision: dict[str, Any],
    ) -> MongoDatasetRecordDocument:
        run_document = self.ensure_run(run_id=run_id)
        record_document = run_document.records[record_id]
        record_document.monitoring_decision = monitoring_decision
        self.save_run(run_document)
        return record_document

    def start_agent_execution(
        self,
        *,
        run_id: str,
        record_id: str,
        agent: str,
        invocation: AgentInvocation,
    ) -> MongoAgentExecutionDocument:
        run_document = self.ensure_run(run_id=run_id)
        record_document = run_document.records[record_id]
        execution_document = self.new_agent_execution_document(
            agent=agent,
            invocation=invocation,
            result=None,
            middleware_audit=[],
        )
        record_document.agent_executions[agent] = execution_document
        self.save_run(run_document)
        return execution_document

    def finish_agent_execution(
        self,
        *,
        run_id: str,
        record_id: str,
        agent: str,
        invocation: AgentInvocation,
        result: dict[str, Any] | None = None,
        middleware_audit: list[dict[str, Any]] | None = None,
    ) -> MongoAgentExecutionDocument:
        run_document = self.ensure_run(run_id=run_id)
        record_document = run_document.records[record_id]
        execution_document = self.new_agent_execution_document(
            agent=agent,
            invocation=invocation,
            result=result,
            middleware_audit=middleware_audit,
        )
        record_document.agent_executions[agent] = execution_document
        self.save_run(run_document)
        return execution_document

    def save_run(self, run_document: MongoRunDocument) -> None:
        self._runs[run_document.run_id] = run_document
        self._collection.replace_one(
            {"_id": run_document.run_id},
            run_document.model_dump(mode="json", by_alias=True),
            upsert=True,
        )


__all__ = ["MongoRepository"]
