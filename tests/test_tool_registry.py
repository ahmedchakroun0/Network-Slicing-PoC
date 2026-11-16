"""The tool registry is the single source of truth: these checks catch drift
between a tool's signature, its governance schema and its safety envelope."""

import pytest

from agents.near_rt_ric_agent import NearRTRICAgent
from agents.non_rt_ric_agent import NonRTRICAgent
from agents.ran_trust_governance import _safety_violation
from tools.registry import (
    PROTECTED_TOOLS,
    REASONING_FIELDS,
    RESOURCE_OF,
    TOOL_OWNERS,
    TOOL_SPECS,
)


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
def test_args_model_matches_tool_signature(spec):
    tool_args = set(spec.tool.args) - set(REASONING_FIELDS)
    assert tool_args == set(spec.args_model.model_fields)


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
def test_every_tool_takes_the_reasoning_fields(spec):
    assert set(REASONING_FIELDS) <= set(spec.tool.args)


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
def test_every_tool_has_a_safety_envelope(spec):
    # Arguments of the right type; the values may be out of range, but the
    # tool must not hit the default-deny branch.
    samples = {int: 0, float: 0.0, str: "x"}
    args = {
        name: samples.get(field.annotation, "x")
        for name, field in spec.args_model.model_fields.items()
    }
    reason = _safety_violation(spec.name, args) or ""
    assert "No safety envelope" not in reason


def test_derived_tables_cover_every_tool():
    names = {spec.name for spec in TOOL_SPECS}
    assert len(names) == len(TOOL_SPECS)
    assert PROTECTED_TOOLS == names
    assert set(RESOURCE_OF) == names
    assert TOOL_OWNERS["near_rt_ric"] | TOOL_OWNERS["non_rt_ric"] == names
    assert not TOOL_OWNERS["near_rt_ric"] & TOOL_OWNERS["non_rt_ric"]


@pytest.mark.parametrize("agent", [NearRTRICAgent, NonRTRICAgent])
def test_agents_get_exactly_their_owned_tools(agent):
    action_tools = {t.name for t in agent.tools if t.name in PROTECTED_TOOLS}
    assert action_tools == TOOL_OWNERS[agent.agent_name]
