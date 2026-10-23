from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ToolExecution(BaseModel):
    """A tool executed by an agent and its result."""

    agent: str
    tool_name: str
    result: Any = None


class WorkingMemory(BaseModel):
    """Shared global network state for all agents."""

    network_status: str = Field(
        default="Network status not yet available",
        description="Current network status summarized by the Monitoring Agent.",
    )

    tools_executed: list[ToolExecution] = Field(
        default_factory=list,
        description="Tools executed by agents and their results.",
    )

    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
