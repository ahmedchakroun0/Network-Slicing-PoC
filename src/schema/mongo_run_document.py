"""Shapes of the MongoDB results store (src/repositories/mongo_repository.py).

Two collections:
  runs     one small document per run: configuration and summary counters.
  records  one document per telemetry record. Every agent step adds to it
           with a targeted update ($set / $push); it is never replaced.

Every fact is stored once. The context passed to the agents (target, slice,
SLA analysis, persistence, slice state) is derivable from `telemetry`, `sla`
and `persistence`, so invocations are stored without it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """The visible text of a model turn (output tokens, not hidden thinking)."""

    text: str | None = None
    forced: bool = Field(
        default=False,
        description="The middleware produced this tool call because the model "
        "did not call an action tool on its own.",
    )


class ActionEntry(BaseModel):
    """One protected tool call: what the RIC agent proposed and what happened."""

    i: int = Field(description="Position in the agent's action list.")
    tool_call_id: str
    at: datetime
    tool: str
    target: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    reasoning: dict[str, Any] | None = Field(
        default=None, description="rationale, expected_effect, evidence_sources, confidence."
    )
    agent_message: AgentMessage | None = Field(
        default=None, description="The model turn that emitted this tool call."
    )
    governance: dict[str, Any] = Field(
        description="approved, failed_stage, reason, checks_passed, and llm_review "
        "(goal_aligned, reasoning_sound, behavior_acceptable, reason) "
        "when the LLM review ran."
    )
    execution: dict[str, Any] = Field(description="status, result, message.")


class KnowledgeEntry(BaseModel):
    """One knowledge lookup of a RIC agent and what it returned."""

    tool_call_id: str
    at: datetime
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str


class AgentRun(BaseModel):
    """One RIC agent invocation inside a record."""

    status: Literal["running", "completed", "rejected", "no_action", "error"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    selected_action: int | None = Field(
        default=None, description="Index into `actions` of the action the result reports."
    )
    actions: list[ActionEntry] = Field(default_factory=list)
    knowledge: list[KnowledgeEntry] = Field(default_factory=list)
    final_message: AgentMessage | None = None
    error: dict[str, Any] | None = None


class RecordDocument(BaseModel):
    """The `records` document of one telemetry record."""

    id: str = Field(alias="_id", description="'<run_id>:<record_id>'.")
    run_id: str
    seq: int = Field(description="Row number in the dataset.")
    status: Literal["received", "decided", "done", "failed"]
    telemetry: dict[str, Any]
    sla: dict[str, Any]
    persistence: dict[str, int] = Field(default_factory=dict)
    monitoring: dict[str, Any] | None = Field(
        default=None,
        description="Effective decision after the deterministic override, plus "
        "llm_output (the model's decision before it, when it was overridden).",
    )
    agents: dict[str, AgentRun] = Field(default_factory=dict)
    llm: dict[str, Any] | None = Field(
        default=None,
        description="Model usage of the record: calls, failed_calls, input_tokens, "
        "output_tokens, calls_without_usage, and the same per caller in by_caller "
        "(monitoring, near_rt_ric, non_rt_ric, governance_review).",
    )
    models: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{model, calls}]: models that answered this record's calls, after "
        "the gateway's fallbacks.",
    )
    error: dict[str, Any] | None = None


class RunDocument(BaseModel):
    """The `runs` document: run configuration and summary counters."""

    id: str = Field(alias="_id")
    created_at: datetime
    finished_at: datetime | None = None
    status: Literal["running", "finished", "stopped"] = "running"
    stop_reason: str | None = Field(
        default=None, description="Why the run ended early (e.g. every model rate-limited).")
    config: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)


__all__ = ["ActionEntry", "AgentMessage", "AgentRun", "KnowledgeEntry", "RecordDocument",
           "RunDocument"]
