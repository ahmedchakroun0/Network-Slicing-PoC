"""Each tool is defined once (a RANControls method); the registry only adds
its owner and resource. These checks catch drift between them;
tests/test_mcp_servers.py checks what the MCP servers serve."""

import inspect

import pytest

from agents.ran_trust_governance import _safety_violation
from ric_tools.controls import RANControls
from ric_tools.registry import (
    ARG_MODELS,
    MCP_SERVER_OF,
    PROTECTED_TOOLS,
    REASONING_FIELDS,
    RESOURCE_OF,
    TOOL_OWNERS,
    TOOL_SPECS,
    specs_of,
)

CONTROL_METHODS = {name for name, _ in inspect.getmembers(RANControls, inspect.iscoroutinefunction)}


def test_every_control_method_is_a_registered_tool():
    assert CONTROL_METHODS == PROTECTED_TOOLS


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
def test_governance_schema_is_the_method_signature(spec):
    params = list(inspect.signature(getattr(RANControls, spec.name)).parameters)[1:]
    assert list(ARG_MODELS[spec.name].model_fields) == params
    assert ARG_MODELS[spec.name].model_config.get("extra") == "forbid"
    # The justification is not a tool argument.
    assert not set(params) & set(REASONING_FIELDS)


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
def test_every_tool_is_described(spec):
    assert inspect.getdoc(getattr(RANControls, spec.name))


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
def test_every_tool_has_a_safety_envelope(spec):
    # Arguments of the right type; the values may be out of range, but the
    # tool must not hit the default-deny branch.
    samples = {int: 0, float: 0.0}
    args = {
        name: samples.get(field.annotation, "x")
        for name, field in ARG_MODELS[spec.name].model_fields.items()
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


@pytest.mark.parametrize("role", ["near_rt_ric", "non_rt_ric"])
def test_every_role_has_a_server_and_tools(role):
    assert MCP_SERVER_OF[role]
    assert {s.name for s in specs_of(role)} == TOOL_OWNERS[role]
