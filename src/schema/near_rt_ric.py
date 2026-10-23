from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


class RetrievedKnowledge(BaseModel):
    content: str
    source: str
    score: float



class RANActionProposal(BaseModel):
    tool_name: Literal[
        "adjust_prb_allocation",
        "optimize_scheduler",
        "optimize_handover",
        "adjust_power_control",
        "adjust_qos_parameters",
        "adjust_link_adaptation",
        "update_slice_resource_policy",
        "update_traffic_steering_policy",
        "update_admission_control_policy",
        "optimize_slice_capacity_policy",
        
    ]

    parameters: dict[str, Any] = Field(default_factory=dict)

    target: str

    rationale: str

    expected_effect: str

    evidence_sources: list[str] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0)
    invocation_context: SkipJsonSchema[dict[str, Any]] = Field(default_factory=dict)


class GovernanceResult(BaseModel):
    approved: bool
    reason: str
    failed_stage: str | None = None
    checks_passed: list[str] = Field(default_factory=list)


class RANToolResult(BaseModel):
    tool_name: str
    status: Literal["executed", "rejected", "error"]
    target: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class NearRTRICResult(BaseModel):
    status: Literal[
        "completed",
        "rejected",
        "no_action",
        "error",
    ]

    proposal: RANActionProposal | None = None

    governance: GovernanceResult | None = None

    execution: RANToolResult | None = None

    retrieved_knowledge: list[RetrievedKnowledge] = Field(default_factory=list)
