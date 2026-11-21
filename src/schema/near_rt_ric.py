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
        
    ]

    parameters: dict[str, Any] = Field(default_factory=dict)

    target: str

    rationale: str

    expected_effect: str

    evidence_sources: list[str] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0)
    invocation_context: SkipJsonSchema[dict[str, Any]] = Field(default_factory=dict)


class LLMReview(BaseModel):
    """The governance LLM's judgement of one action, kept for evaluation."""

    goal_aligned: bool
    reasoning_sound: bool
    behavior_acceptable: bool
    reason: str


class GovernanceResult(BaseModel):
    approved: bool
    reason: str
    failed_stage: str | None = None
    checks_passed: list[str] = Field(default_factory=list)
    # Present when the LLM review stage ran (stages 1-4 passed).
    llm_review: LLMReview | None = None


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
