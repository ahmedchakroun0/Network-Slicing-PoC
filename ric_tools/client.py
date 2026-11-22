"""The RIC agents' action tools, loaded from their MCP server.

    async with role_tools("near_rt_ric") as tools:   # through LiteLLM
        agent = NearRTRICAgent(action_tools=tools, ...)

The MCP tools take only the arguments the RAN uses. The model must also
justify each call for governance, so every tool the model sees gets the
REASONING_FIELDS added to its schema; they are stripped before the MCP call
(the middleware has already moved them into the governance proposal).

The agents reach the MCP servers through the LiteLLM gateway, at
{LITELLM_API_BASE}/{server}/mcp with the LiteLLM key. Each MCP tool becomes
a plain LangChain tool: the model sees it as a function tool, and the tool
call goes through RANAuthorizationMiddleware (governance) BEFORE the MCP
call is made. LiteLLM is only the MCP endpoint here: tools are never sent
to a completion as `{"type": "mcp"}`, which would let the gateway execute
them and skip governance.

LiteLLM prefixes tool names with their server ("near_rt-adjust_prb_allocation").
The LangChain tool keeps the registry name, which the middleware and the
audits use; the MCP call uses the name as listed. A listed tool the role does
not own is not offered to the agent (fail closed).
"""

from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx2
from langchain_core.tools import StructuredTool, ToolException
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from ric_tools.registry import MCP_SERVER_OF, REASONING_FIELDS, TOOL_OWNERS, AgentRole

logger = logging.getLogger(__name__)

# Same default as LiteLLM's MCP_TOOL_PREFIX_SEPARATOR.
PREFIX_SEPARATOR = os.environ.get("MCP_TOOL_PREFIX_SEPARATOR", "-")


def registry_name(listed: str, server: str) -> str:
    """The registry name of a tool listed by the gateway (prefix removed)."""
    prefix = f"{server}{PREFIX_SEPARATOR}"
    return listed.removeprefix(prefix)


# What the model sees of the justification fields (not sent to the MCP server).
REASONING_SCHEMA: dict[str, dict[str, Any]] = {
    "rationale": {"type": "string", "description": (
        "Why this tool and these exact values: quote measured and budget of the violated "
        "KPIs.")},
    "expected_effect": {"type": "string",
                        "description": "Which KPI should improve, and how."},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1,
                   "description": "Your confidence that the action fixes the violation."},
    "evidence_sources": {"type": "array", "items": {"type": "string"},
                         "description": "The knowledge snippets and context facts you used."},
}
REQUIRED_REASONING = ("rationale", "expected_effect", "confidence")
assert set(REASONING_SCHEMA) == set(REASONING_FIELDS)


def model_schema(input_schema: dict[str, Any]) -> dict[str, Any]:
    """The tool's MCP input schema plus the justification fields."""
    schema = copy.deepcopy(input_schema)
    schema.setdefault("properties", {}).update(copy.deepcopy(REASONING_SCHEMA))
    schema["required"] = [*schema.get("required", []), *REQUIRED_REASONING]
    return schema


def _result_text(result: Any) -> str:
    return "\n".join(getattr(block, "text", "") for block in result.content or [])


def _wrap(client: Client, listed: Any, name: str) -> StructuredTool:
    async def call(**arguments: Any) -> str:
        action = {k: v for k, v in arguments.items() if k not in REASONING_FIELDS}
        result = await client.call_tool(listed.name, action)
        text = _result_text(result)
        if result.is_error:
            raise ToolException(text or f"{name} failed")
        if result.structured_content is not None:
            return json.dumps(result.structured_content)
        return text

    return StructuredTool(
        name=name,
        description=listed.description or "",
        args_schema=model_schema(listed.input_schema),
        coroutine=call,
    )


async def load_tools(client: Client, role: AgentRole) -> list[StructuredTool]:
    """The role's action tools listed by an open MCP client."""
    server = MCP_SERVER_OF[role]
    owned = TOOL_OWNERS[role]
    tools = []
    for listed in (await client.list_tools()).tools:
        name = registry_name(listed.name, server)
        if name not in owned:
            logger.warning("MCP server %s lists %r, which %s does not own: not offered.",
                           server, listed.name, role)
            continue
        tools.append(_wrap(client, listed, name))
    missing = owned - {t.name for t in tools}
    if missing:
        logger.warning("MCP server %s does not list %s for %s.", server, sorted(missing), role)
    return tools


@asynccontextmanager
async def role_tools(
    role: AgentRole,
    *,
    client: Client | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> AsyncIterator[list[StructuredTool]]:
    """Open the role's MCP server through LiteLLM (or use an open `client`,
    e.g. an in-process one in tests) and yield its action tools."""
    async with AsyncExitStack() as stack:
        if client is None:
            base = (base_url or os.environ["LITELLM_API_BASE"]).rstrip("/")
            key = api_key or os.environ["LITELLM_API_KEY"]
            url = f"{base}/{MCP_SERVER_OF[role]}/mcp"
            http = await stack.enter_async_context(httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {key}"}, timeout=timeout_s))
            client = await stack.enter_async_context(
                Client(streamable_http_client(url, http_client=http)))
        yield await load_tools(client, role)
