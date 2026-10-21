from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schema.monitoring import AgentInvocation


class MongoMiddlewareAuditDocument(BaseModel):
    """One middleware audit entry captured for a protected tool call."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    tool_call_id: str
    proposal: dict[str, Any] | None = None
    governance: dict[str, Any]
    execution: dict[str, Any]


class MongoAgentExecutionDocument(BaseModel):
    """All persisted data for one specialized agent invocation."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    agent: Literal["near_rt_ric", "non_rt_ric"]
    invocation: AgentInvocation
    result: dict[str, Any] | None = None
    middleware_audit: list[MongoMiddlewareAuditDocument] = Field(default_factory=list)


class MongoDatasetRecordDocument(BaseModel):
    """Mongo document fragment for one dataset row / telemetry record."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    record_id: str
    telemetry_record: dict[str, Any]
    ground_truth: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] = Field(default_factory=dict)
    computed_analysis: dict[str, Any] = Field(default_factory=dict)
    monitoring_decision: dict[str, Any] | None = None
    agent_executions: dict[str, MongoAgentExecutionDocument] = Field(
        default_factory=dict
    )
    error: dict[str, Any] | None = None


class MongoRunDocument(BaseModel):
    """Top-level Mongo document for one full dataset execution run."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str = Field(alias="_id")
    run_id: str
    summary: dict[str, Any] = Field(default_factory=dict)
    records: dict[str, MongoDatasetRecordDocument] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.id != self.run_id:
            raise ValueError("MongoRunDocument id and run_id must match.")

    def upsert_record(self, record: MongoDatasetRecordDocument) -> None:
        self.records[record.record_id] = record

    def get_record(self, record_id: str) -> MongoDatasetRecordDocument | None:
        return self.records.get(record_id)

    def as_mongo_document(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json")


__all__ = [
    "MongoMiddlewareAuditDocument",
    "MongoAgentExecutionDocument",
    "MongoDatasetRecordDocument",
    "MongoRunDocument",
]
