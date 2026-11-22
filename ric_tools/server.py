"""MCP servers of the RIC action tools, one per role.

    near_rt  the six Near-RT RIC controls (E2-style)
    non_rt   the Non-RT RIC slice policy (A1-style)

Each server serves the RANControls methods (controls.py) its role owns,
under their own names, arguments and descriptions. An emulator refusal
(out-of-range value, unknown cell) is returned as an MCP tool error.

The servers do not authorize anything: the RIC agents' governance middleware
has already approved every call that reaches them, and the emulator's safety
envelope is the last line of defense. They listen on 127.0.0.1 and are
reached by the agents through the LiteLLM gateway.
"""

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ran_emulator.client import EmulatorClient, EmulatorError
from ric_tools.controls import RANControls
from ric_tools.registry import MCP_SERVER_OF, AgentRole, specs_of

ROLE_OF_SERVER: dict[str, AgentRole] = {server: role for role, server in MCP_SERVER_OF.items()}


def _refusals_as_tool_errors(
    method: Callable[..., Awaitable[dict[str, Any]]],
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """The method unchanged (signature, docstring), except that an emulator
    refusal becomes an MCP tool error the agent can read."""

    @functools.wraps(method)
    async def tool(**arguments: Any) -> dict[str, Any]:
        try:
            return await method(**arguments)
        except EmulatorError as exc:
            raise ToolError(
                f"the RAN refused the control ({exc.status_code}): {exc.detail}") from exc

    return tool


def build_server(role: AgentRole, emulator: EmulatorClient) -> MCPServer:
    """The MCP server of one RIC role: exactly the tools the role owns."""
    controls = RANControls(emulator)
    server = MCPServer(
        name=f"ric-{MCP_SERVER_OF[role]}",
        instructions=inspect.getdoc(RANControls),
    )
    for spec in specs_of(role):
        method = getattr(controls, spec.name)
        server.tool(name=spec.name, description=inspect.getdoc(method))(
            _refusals_as_tool_errors(method))
    return server
