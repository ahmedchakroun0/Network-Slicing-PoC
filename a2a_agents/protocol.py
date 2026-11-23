"""A2A plumbing shared by the agent servers: JSON payloads, executor, app.

Every skill of this system takes one JSON object and returns one JSON object.
The payload travels as a text part with media type application/json (exact
JSON: a protobuf data part would turn every integer into a float). A data
part is also accepted on input, for clients that send one.

A skill handler is `async (payload, metadata) -> result`. The executor runs it
inside the A2A task lifecycle: working -> completed with one artifact holding
the result, or failed with the error text.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Any

from a2a.helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Part
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.protobuf.json_format import MessageToDict
from langsmith.run_helpers import get_current_run_tree, tracing_context

logger = logging.getLogger(__name__)

JSON_MEDIA_TYPE = "application/json"
CARD_PATH = "/.well-known/agent-card.json"
TRACE_KEY = "langsmith_headers"      # message metadata: parent trace of the caller

Handler = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]


# --------------------------------------------------------------------- payloads

def json_part(payload: dict[str, Any]) -> Part:
    return Part(text=json.dumps(payload), media_type=JSON_MEDIA_TYPE)


def read_json(parts: Sequence[Part]) -> dict[str, Any]:
    """The JSON object of the first JSON text part (or data part)."""
    for part in parts:
        if part.text:
            value = json.loads(part.text)
        elif part.HasField("data"):
            value = MessageToDict(part.data)
        else:
            continue
        if not isinstance(value, dict):
            raise ValueError("the payload must be a JSON object")
        return value
    raise ValueError("the message has no JSON payload")


def trace_metadata() -> dict[str, Any]:
    """Headers of the current LangSmith run, so the callee nests under it."""
    run = get_current_run_tree()
    return {TRACE_KEY: run.to_headers()} if run is not None else {}


@contextlib.contextmanager
def continue_trace(metadata: dict[str, Any]) -> Iterator[None]:
    headers = metadata.get(TRACE_KEY)
    if headers:
        with tracing_context(parent=dict(headers)):
            yield
    else:
        yield


# ------------------------------------------------------------------ the executor

class JsonSkillExecutor(AgentExecutor):
    """Runs one JSON skill handler per A2A task."""

    def __init__(self, handler: Handler, artifact_name: str):
        self.handler = handler
        self.artifact_name = artifact_name

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        try:
            payload = read_json(context.message.parts)
            metadata = MessageToDict(context.message.metadata)
            with continue_trace(metadata):
                result = await self.handler(payload, metadata)
        except Exception as exc:
            logger.exception("%s failed", self.artifact_name)
            await updater.failed(updater.new_agent_message(
                [Part(text=f"{type(exc).__name__}: {exc}")]))
            return
        await updater.add_artifact([json_part(result)], name=self.artifact_name)
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.task_id and context.context_id:
            await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


# --------------------------------------------------------------------- the app

def agent_card(*, name: str, description: str, url: str, skill: AgentSkill,
               version: str = "0.1.0") -> AgentCard:
    return AgentCard(
        name=name, description=description, version=version,
        supported_interfaces=[AgentInterface(url=url, protocol_binding="JSONRPC",
                                             protocol_version="1.0")],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=[JSON_MEDIA_TYPE], default_output_modes=[JSON_MEDIA_TYPE],
        skills=[skill],
    )


def require_bearer(app: FastAPI, token: str | None,
                   open_paths: frozenset[str] = frozenset({CARD_PATH, "/health"})) -> None:
    """Every route except discovery and health needs `Authorization: Bearer <token>`.
    No token configured: open (local development)."""
    if not token:
        return
    expected = f"Bearer {token}"

    @app.middleware("http")
    async def _check(request: Request, call_next):
        if request.url.path not in open_paths and not hmac.compare_digest(
                request.headers.get("authorization", ""), expected):
            return JSONResponse({"detail": "missing or invalid bearer token"}, status_code=401)
        return await call_next(request)


def build_a2a_app(card: AgentCard, handler: Handler, *, artifact_name: str,
                  token: str | None = None, lifespan: Any = None) -> FastAPI:
    """A FastAPI app serving the agent card and the A2A JSON-RPC endpoint at /.

    v0.3 compatibility is on: the LiteLLM A2A gateway forwards v0.3 requests.
    `lifespan` opens what the handler needs (MCP session, discovery) at start-up.
    """
    request_handler = DefaultRequestHandler(
        agent_executor=JsonSkillExecutor(handler, artifact_name),
        task_store=InMemoryTaskStore(), agent_card=card)
    app = FastAPI(title=card.name, lifespan=lifespan)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(request_handler, rpc_url="/",
                                             enable_v0_3_compat=True),
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "agent": card.name}

    require_bearer(app, token)
    return app
