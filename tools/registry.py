"""Single registry of the protected RAN action tools.

Every fact about an action tool lives here once:

    tool         the LangChain tool the RIC agent calls
    owner        the RIC agent allowed to call it (role binding, governance stage 2)
    resource     the network resource it controls (one executed change per
                 resource per invocation, enforced by the middleware)
    args_model   the exact argument schema governance validates (stage 1)

Everything else is derived from TOOL_SPECS: PROTECTED_TOOLS, TOOL_OWNERS,
RESOURCE_OF, ARG_MODELS and each agent's action tool list.

To add an action tool: write the tool, add its args model and its ToolSpec
here, and give it a safety envelope in agents/ran_trust_governance.py
(governance denies a tool without one). tests/test_tool_registry.py checks
that each args model matches its tool's signature.
"""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tools.near_rt_ric_tools import (
    adjust_link_adaptation,
    adjust_power_control,
    adjust_prb_allocation,
    adjust_qos_parameters,
    optimize_handover,
    optimize_scheduler,
)
from tools.non_rt_ric_tools import update_slice_resource_policy

AgentRole = Literal["near_rt_ric", "non_rt_ric"]

# Arguments every protected tool takes besides its own action parameters.
# The middleware moves them into the proposal; they are not action arguments.
REASONING_FIELDS = ("rationale", "expected_effect", "evidence_sources", "confidence")


# --------------------------------------------------------------------------
# Argument schemas (governance stage 1). extra="forbid": an argument the
# tool does not declare is rejected, not silently dropped.
# --------------------------------------------------------------------------


class _BaseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1)


class PRBArgs(_BaseArgs):
    slice_id: str = Field(min_length=1)
    prb_percentage: int


class SchedulerArgs(_BaseArgs):
    slice_id: str = Field(min_length=1)
    # A plain str here (not the Scheduler literal): an unknown scheduler is
    # rejected by the safety stage, with the list of allowed values.
    scheduler: str = Field(min_length=1)


class HandoverArgs(_BaseArgs):
    handover_margin_db: float


class PowerArgs(_BaseArgs):
    power_offset_db: float


class QoSArgs(_BaseArgs):
    slice_id: str = Field(min_length=1)
    priority_level: int
    packet_delay_budget_ms: float
    guaranteed_bitrate_mbps: float


class LinkAdaptationArgs(_BaseArgs):
    slice_id: str = Field(min_length=1)
    mcs_offset: int
    target_bler: float


class SlicePolicyArgs(_BaseArgs):
    slice_id: str = Field(min_length=1)
    min_prb_ratio_pct: int
    max_prb_ratio_pct: int


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    tool: Any
    owner: AgentRole
    resource: str
    args_model: type[BaseModel]

    @property
    def name(self) -> str:
        return self.tool.name


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(adjust_prb_allocation, "near_rt_ric", "prb_allocation", PRBArgs),
    ToolSpec(optimize_scheduler, "near_rt_ric", "scheduler", SchedulerArgs),
    ToolSpec(optimize_handover, "near_rt_ric", "handover", HandoverArgs),
    ToolSpec(adjust_power_control, "near_rt_ric", "power", PowerArgs),
    ToolSpec(adjust_qos_parameters, "near_rt_ric", "qos", QoSArgs),
    ToolSpec(adjust_link_adaptation, "near_rt_ric", "link_adaptation", LinkAdaptationArgs),
    ToolSpec(update_slice_resource_policy, "non_rt_ric", "prb_policy", SlicePolicyArgs),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}

PROTECTED_TOOLS: frozenset[str] = frozenset(TOOLS_BY_NAME)
ARG_MODELS: dict[str, type[BaseModel]] = {s.name: s.args_model for s in TOOL_SPECS}
RESOURCE_OF: dict[str, str] = {s.name: s.resource for s in TOOL_SPECS}


def tools_of(owner: AgentRole) -> list[Any]:
    """The action tools a RIC agent owns, in registry order."""
    return [spec.tool for spec in TOOL_SPECS if spec.owner == owner]


TOOL_OWNERS: dict[str, frozenset[str]] = {
    owner: frozenset(spec.name for spec in TOOL_SPECS if spec.owner == owner)
    for owner in ("near_rt_ric", "non_rt_ric")
}
NEAR_RT_TOOLS = TOOL_OWNERS["near_rt_ric"]
NON_RT_POLICY_TOOLS = TOOL_OWNERS["non_rt_ric"]
