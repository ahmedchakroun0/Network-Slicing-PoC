"""The MCP servers act on the RAN emulator, and the client adapter turns their
tools into plain LangChain tools with registry names (as LiteLLM lists them,
prefixed with the server name) and the justification fields governance needs."""

import inspect
import json

import httpx
import pytest
from langchain_core.tools import ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from mcp import Client
from mcp.server.mcpserver import MCPServer

from ran_emulator.api import create_app
from ran_emulator.client import EmulatorClient
from ric_tools.client import REASONING_SCHEMA, load_tools, registry_name, role_tools
from ric_tools.controls import RANControls
from ric_tools.registry import ARG_MODELS, MCP_SERVER_OF, TOOL_OWNERS, TOOL_SPECS
from ric_tools.server import build_server

REASONING = {
    "rationale": "eMBB throughput is below budget.",
    "expected_effect": "Higher rate.",
    "confidence": 0.8,
}

CALLS = {
    "adjust_prb_allocation": {"slice_id": "eMBB", "prb_percentage": 72},
    "optimize_scheduler": {"scheduler": "proportional_fair"},
    "optimize_handover": {"handover_margin_db": 2.0},
    "adjust_power_control": {"power_offset_db": 3.0},
    "adjust_qos_parameters": {"slice_id": "URLLC", "priority_level": 2},
    "adjust_link_adaptation": {"slice_id": "URLLC", "mcs_offset": -2, "target_bler": 0.01},
    "update_slice_resource_policy": {
        "slice_id": "eMBB",
        "min_prb_ratio_pct": 20,
        "max_prb_ratio_pct": 90,
    },
}


@pytest.fixture
def emulator(bank, make_scenario):
    app = create_app(bank, make_scenario(tr=9))
    return EmulatorClient(transport=httpx.ASGITransport(app=app), base_url="http://emu")


@pytest.mark.parametrize("role", ["near_rt_ric", "non_rt_ric"])
async def test_each_server_lists_exactly_its_role_tools(emulator, role):
    async with Client(build_server(role, emulator)) as client:
        listed = {t.name for t in (await client.list_tools()).tools}
    assert listed == TOOL_OWNERS[role]


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
async def test_served_schema_is_the_control_method(emulator, spec):
    """The MCP tool takes exactly the method's arguments (what governance
    validates), with the method's docstring as description."""
    async with Client(build_server(spec.owner, emulator)) as client:
        (listed,) = [t for t in (await client.list_tools()).tools if t.name == spec.name]
    expected = ARG_MODELS[spec.name].model_json_schema()
    assert listed.input_schema["properties"].keys() == expected["properties"].keys()
    assert set(listed.input_schema["required"]) == set(expected["required"])
    assert listed.description == inspect.getdoc(getattr(RANControls, spec.name))


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
async def test_every_tool_changes_the_emulator(emulator, spec):
    async with Client(build_server(spec.owner, emulator)) as client:
        result = await client.call_tool(spec.name, {"target": "bs2", **CALLS[spec.name]})
    assert not result.is_error, result.content
    payload = json.loads(result.content[0].text)
    assert {"previous", "current", "effective_step"} <= payload.keys()
    actions = await emulator._call("GET", "/sim/actions")
    assert len(actions) == 1


async def test_prb_reports_requested_and_applied_share(emulator):
    async with Client(build_server("near_rt_ric", emulator)) as client:
        result = await client.call_tool(
            "adjust_prb_allocation", {"target": "bs2", "slice_id": "eMBB", "prb_percentage": 70}
        )
    payload = json.loads(result.content[0].text)
    assert payload["requested_pct"] == 70
    assert payload["applied_pct"] == payload["current"]["eMBB"] * 2  # 50 PRBs


async def test_refused_control_is_an_mcp_error_and_a_violation(emulator):
    async with Client(build_server("near_rt_ric", emulator)) as client:
        result = await client.call_tool(
            "adjust_power_control", {"target": "bs2", "power_offset_db": 20.0}
        )
    assert result.is_error
    assert "power_offset_db must be within" in result.content[0].text
    assert (await emulator.violations())[0]["control"] == "power"


# ---------------------------------------------------------------- client side


def test_registry_name_strips_the_litellm_prefix():
    assert registry_name("near_rt-adjust_prb_allocation", "near_rt") == "adjust_prb_allocation"
    assert registry_name("adjust_prb_allocation", "near_rt") == "adjust_prb_allocation"


async def test_prefixed_tools_get_registry_names_and_plain_function_schemas(emulator):
    """As LiteLLM lists them: '<server>-<tool>'. A tool the role does not own
    is not offered (fail closed)."""
    server = MCPServer(name="gateway-like")
    for name in ("near_rt-adjust_power_control", "near_rt-reboot_cell"):
        server.tool(name=name, description=name)(lambda target: {"ok": target})

    async with Client(server) as client:
        tools = await load_tools(client, "near_rt_ric")

    assert [t.name for t in tools] == ["adjust_power_control"]
    # The model sees a plain function tool; LiteLLM never executes it itself.
    assert convert_to_openai_tool(tools[0])["type"] == "function"


async def test_model_sees_the_justification_fields_but_the_server_does_not(emulator):
    async with (
        Client(build_server("near_rt_ric", emulator)) as client,
        role_tools("near_rt_ric", client=client) as tools,
    ):
        prb = next(t for t in tools if t.name == "adjust_prb_allocation")
        schema = prb.args_schema
        assert set(REASONING_SCHEMA) <= schema["properties"].keys()
        assert {"rationale", "expected_effect", "confidence"} <= set(schema["required"])
        # The call carries the justification; the MCP server never gets it.
        sent = []
        call_tool = client.call_tool

        async def spy(name, arguments):
            sent.append(arguments)
            return await call_tool(name, arguments)

        client.call_tool = spy
        out = json.loads(
            await prb.ainvoke(
                {
                    "target": "bs2",
                    "slice_id": "eMBB",
                    "prb_percentage": 72,
                    **REASONING,
                    "evidence_sources": ["slice_state"],
                }
            )
        )
    assert sent == [{"target": "bs2", "slice_id": "eMBB", "prb_percentage": 72}]
    assert out["requested_pct"] == 72


async def test_tool_calls_the_listed_name_and_raises_on_error(emulator):
    async with (
        Client(build_server("near_rt_ric", emulator)) as client,
        role_tools("near_rt_ric", client=client) as tools,
    ):
        power = next(t for t in tools if t.name == "adjust_power_control")
        out = json.loads(
            await power.ainvoke({"target": "bs2", "power_offset_db": 3.0, **REASONING})
        )
        assert out["current"] == {"power_offset_db": 3.0}
        with pytest.raises(ToolException, match="power_offset_db must be within"):
            await power.ainvoke({"target": "bs2", "power_offset_db": 20.0, **REASONING})


def test_server_names_match_litellm_config():
    import yaml

    config = yaml.safe_load(open("litellm_config.yaml"))
    servers = config["mcp_servers"]
    for role, name in MCP_SERVER_OF.items():
        assert set(servers[name]["allowed_tools"]) == TOOL_OWNERS[role]
