"""A2A client of the JSON skills: discovery by Agent Card, one task per call."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx
from a2a.client import A2ACardResolver, A2AClientError, Client, ClientConfig, ClientFactory
from a2a.helpers import get_text_parts
from a2a.types import AgentCard, Message, Role, SendMessageRequest, TaskState

from a2a_agents.protocol import json_part, read_json, trace_metadata

logger = logging.getLogger(__name__)


class A2ACallError(RuntimeError):
    """The remote agent could not be reached or its task did not complete."""


class A2AJsonClient:
    """Client of one remote agent, found from its base URL (where its card is).

    `token` is sent as a bearer token on every call (the agent's own token, or
    the gateway key when the base URL is the LiteLLM A2A gateway).
    """

    def __init__(self, base_url: str, *, token: str | None = None, timeout_s: float = 300.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.AsyncClient(headers=headers, timeout=timeout_s, transport=transport)
        self._card: AgentCard | None = None
        self._client: Client | None = None

    async def card(self) -> AgentCard:
        if self._card is None:
            try:
                self._card = await A2ACardResolver(self._http, self.base_url).get_agent_card()
            except Exception as exc:
                raise A2ACallError(f"no agent card at {self.base_url}: {exc}") from exc
        return self._card

    async def skill_ids(self) -> list[str]:
        return [s.id for s in (await self.card()).skills]

    async def call(self, payload: dict[str, Any],
                   metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send the payload as one task; return the JSON of its first artifact."""
        if self._client is None:
            config = ClientConfig(streaming=False, httpx_client=self._http)
            self._client = ClientFactory(config).create(await self.card())
        message = Message(message_id=uuid.uuid4().hex, role=Role.ROLE_USER,
                          parts=[json_part(payload)],
                          metadata={**trace_metadata(), **(metadata or {})})
        name = self._card.name if self._card else self.base_url
        try:
            task = None
            async for event in self._client.send_message(SendMessageRequest(message=message)):
                if event.HasField("task"):
                    task = event.task
        except (A2AClientError, httpx.HTTPError) as exc:
            raise A2ACallError(f"{name}: {type(exc).__name__}: {exc}") from exc
        if task is None:
            raise A2ACallError(f"{name} answered without a task")
        state = task.status.state
        if state != TaskState.TASK_STATE_COMPLETED:
            why = " ".join(get_text_parts(task.status.message.parts)) or "no reason given"
            raise A2ACallError(f"{name}: task {TaskState.Name(state)}: {why}")
        if not task.artifacts:
            raise A2ACallError(f"{name}: completed without an artifact")
        return read_json(task.artifacts[0].parts)

    async def aclose(self) -> None:
        await self._http.aclose()


async def discover(base_urls: list[str], *, tokens: dict[str, str | None] | None = None,
                   timeout_s: float = 300.0,
                   transport: httpx.AsyncBaseTransport | None = None) -> dict[str, A2AJsonClient]:
    """Map skill id -> client, from the agents' cards (no endpoint is hard-coded).

    `tokens` gives the bearer token per skill id; the card itself is public.
    """
    found: dict[str, A2AJsonClient] = {}
    for url in base_urls:
        probe = A2AJsonClient(url, timeout_s=timeout_s, transport=transport)
        try:
            skills = await probe.skill_ids()
        finally:
            await probe.aclose()
        for skill in skills:
            if skill in found:
                raise ValueError(f"two agents declare the skill {skill!r}")
            found[skill] = A2AJsonClient(url, token=(tokens or {}).get(skill),
                                         timeout_s=timeout_s, transport=transport)
            logger.info("A2A: skill %s at %s", skill, url)
    return found
