"""Who owns each RAN action tool, and what it controls.

The tools themselves (arguments and description) are the methods of
RANControls (controls.py). The registry adds what governance needs:

    owner     the RIC agent allowed to call it (role binding, governance stage 2)
    resource  the network resource it changes (one executed change per
              resource per invocation, enforced by the middleware)

and derives from the methods the exact argument schema governance validates
(ARG_MODELS, stage 1). tests/test_tool_registry.py checks they agree.

To add an action tool: write its RANControls method (and its emulator
control), add its ToolSpec here, and give it a safety envelope in
src/schema/limits.py + agents/ran_trust_governance.py (governance denies a
tool without one).
"""

import inspect
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, create_model

from ric_tools.controls import RANControls

AgentRole = Literal["near_rt_ric", "non_rt_ric"]

# The model's justification of each call. Not a tool argument: the client
# adds these fields to what the model sees, the middleware moves them into the
# governance proposal, and they never reach the MCP server.
REASONING_FIELDS = ("rationale", "expected_effect", "evidence_sources", "confidence")

# MCP server of each role: its name in litellm_config.yaml and its URL path
# on the gateway ({LITELLM_API_BASE}/{server}/mcp).
MCP_SERVER_OF: dict[AgentRole, str] = {"near_rt_ric": "near_rt", "non_rt_ric": "non_rt"}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    owner: AgentRole
    resource: str


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("adjust_prb_allocation", "near_rt_ric", "prb_allocation"),
    ToolSpec("optimize_scheduler", "near_rt_ric", "scheduler"),
    ToolSpec("optimize_handover", "near_rt_ric", "handover"),
    ToolSpec("adjust_power_control", "near_rt_ric", "power"),
    ToolSpec("adjust_qos_parameters", "near_rt_ric", "qos"),
    ToolSpec("adjust_link_adaptation", "near_rt_ric", "link_adaptation"),
    ToolSpec("update_slice_resource_policy", "non_rt_ric", "prb_policy"),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}
PROTECTED_TOOLS: frozenset[str] = frozenset(TOOLS_BY_NAME)
RESOURCE_OF: dict[str, str] = {s.name: s.resource for s in TOOL_SPECS}
TOOL_OWNERS: dict[str, frozenset[str]] = {
    owner: frozenset(spec.name for spec in TOOL_SPECS if spec.owner == owner)
    for owner in ("near_rt_ric", "non_rt_ric")
}


def specs_of(owner: AgentRole) -> list[ToolSpec]:
    """The action tools a RIC agent owns, in registry order."""
    return [spec for spec in TOOL_SPECS if spec.owner == owner]


def _args_model(name: str) -> type[BaseModel]:
    """The tool's arguments, from its RANControls method. extra="forbid": an
    argument the tool does not declare is rejected, not silently dropped."""
    params = list(inspect.signature(getattr(RANControls, name)).parameters.values())[1:]
    fields = {
        p.name: (p.annotation, ... if p.default is inspect.Parameter.empty else p.default)
        for p in params
    }
    return create_model(f"{name}_args", __config__=ConfigDict(extra="forbid"), **fields)


ARG_MODELS: dict[str, type[BaseModel]] = {s.name: _args_model(s.name) for s in TOOL_SPECS}
