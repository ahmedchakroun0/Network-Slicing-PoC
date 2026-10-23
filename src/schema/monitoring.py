from typing import Literal, Any

from pydantic import BaseModel, ConfigDict, Field


class TelemetryRecord(BaseModel):
    """
    One network telemetry observation coming from the dataset.

    The field names intentionally correspond to the original
    CSV columns so that the dataset can be passed to the agent
    without changing its semantic structure.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    use_case_type: str = Field(alias="Use Case Type")

    packet_loss_budget: float | None = Field(default=None, alias="Packet Loss Budget")

    latency_budget_ns: float | None = Field(default=None, alias="Latency Budget (ns)")

    jitter_budget_ns: float | None = Field(default=None, alias="Jitter Budget (ns)")

    data_rate_budget_gbps: float | None = Field(
        default=None, alias="Data Rate Budget (Gbps)"
    )

    required_mobility: str | None = Field(default=None, alias="Required Mobility")

    required_connectivity: str | None = Field(
        default=None, alias="Required Connectivity"
    )

    slice_available_transfer_rate_gbps: float | None = Field(
        default=None, alias="Slice Available Transfer Rate (Gbps)"
    )

    slice_latency_ns: float | None = Field(default=None, alias="Slice Latency (ns)")

    slice_packet_loss: float | None = Field(default=None, alias="Slice Packet Loss")

    slice_jitter_ns: float | None = Field(default=None, alias="Slice Jitter (ns)")

    slice_type: str | None = Field(default=None, alias="Slice Type")

    slice_handover: str | None = Field(default=None, alias="Slice Handover")

    # Optional cell / base-station identifier. Datasets without it fall back
    # to a single default target in scripts/run_dataset.py.
    cell_id: str | None = Field(default=None, alias="Cell ID")

    record_id: str | None = None


class KPIStatus(BaseModel):
    measured: float | None = None
    budget: float | None = None
    status: Literal["ok", "violation", "unavailable"]


class SLAAnalysis(BaseModel):
    latency: KPIStatus
    jitter: KPIStatus
    packet_loss: KPIStatus
    throughput: KPIStatus

    violations: list[str] = Field(default_factory=list)
    unavailable_metrics: list[str] = Field(default_factory=list)


AgentName = Literal[
    "near_rt_ric",
    "non_rt_ric",
]


class AgentInvocation(BaseModel):
    """
    Instructions produced by the Monitoring Agent for a
    downstream specialized agent.
    """

    agent: AgentName

    responsibility: str = Field(
        description=("What this agent is responsible for in the current situation.")
    )

    objective: str = Field(
        description=(
            "The specific objective that the selected agent should accomplish."
        )
    )

    priority: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]

    reason: str = Field(description=("Why the Monitoring Agent selected this agent."))
    context: dict[str, Any] = Field(
        default_factory=dict,
        description=("Additional context information for the selected agent."),
    )


class MonitoringDecision(BaseModel):
    """
    Decision produced by the Monitoring Agent.
    """

    situation: Literal[
        "normal",
        "warning",
        "critical",
    ]

    severity: Literal[
        "none",
        "low",
        "medium",
        "high",
    ]

    global_situation: str = Field(
        description=("Concise description of the overall network and slice situation.")
    )

    action_required: bool

    sla_analysis: SLAAnalysis

    selected_agents: list[AgentInvocation] = Field(default_factory=list)

    reason: str = Field(
        description=("Overall explanation of the orchestration decision.")
    )

    record_id: str | None = None
