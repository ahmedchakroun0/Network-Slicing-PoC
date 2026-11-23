from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from src.schema.sla import SliceType

# MAC schedulers of the ColO-RAN dataset (scheduling_policy 0, 1, 2). The
# telemetry, the optimize_scheduler tool and governance share this list.
Scheduler = Literal["round_robin", "waterfilling", "proportional_fair"]


class TelemetryRecord(BaseModel):
    """
    Slice-level telemetry of one cell over one time window, in native
    ColO-RAN units (scripts/prepare_coloran.py builds it from the per-UE
    KPM reports). The field names are the CSV column names, so a prepared
    row validates as-is.

    Measurements only: SLA budgets live in src/schema/sla.py.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str
    cell_id: str = Field(description="Base station, e.g. 'bs3'.")
    slice_id: SliceType
    run: str = Field(description="Source experiment, e.g. 'sched0/tr6/exp1'.")
    window_start_s: float = Field(description="Window start, seconds after the run started.")
    window_s: float

    # Configuration in force during the window.
    slice_prb: int = Field(description="PRBs allocated to this slice (out of cell_prb).")
    cell_prb: int = Field(description="PRBs of the cell (10 MHz carrier: 50).")
    scheduler: Scheduler

    # Measurements, aggregated over the slice's UEs and the window.
    active_ues: int
    dl_throughput_mbps: float = Field(description="Slice downlink throughput (sum over UEs).")
    dl_throughput_per_ue_mbps: float
    dl_buffer_bytes: float = Field(description="Mean downlink backlog per UE.")
    queueing_delay_ms: float | None = Field(
        default=None,
        description="Backlog / delivery rate per UE; the window length when a "
        "backlog was not served at all.",
    )
    prb_utilisation: float = Field(description="Granted / allocated PRBs, 0-1.")
    cell_spare_prb: float = Field(
        description="PRBs allocated to the cell's OTHER slices but left unused."
    )
    dl_cqi: float
    dl_mcs: float
    ul_sinr_db: float | None = Field(
        default=None, description="Mean over reports with uplink activity."
    )


class KPIStatus(BaseModel):
    measured: float | None = None
    budget: float | None = None
    unit: str | None = None
    bound: Literal["min", "max"] | None = None
    status: Literal["ok", "violation", "unavailable"]


class SLAAnalysis(BaseModel):
    slice_id: str | None = None
    kpis: dict[str, KPIStatus] = Field(default_factory=dict)

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
        description=("The specific objective that the selected agent should accomplish.")
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

    # Computed by code and set after the LLM call; the LLM never produces it.
    sla_analysis: SkipJsonSchema[SLAAnalysis | None] = None

    selected_agents: list[AgentInvocation] = Field(default_factory=list)

    reason: str = Field(description=("Overall explanation of the orchestration decision."))

    record_id: str | None = None

    # Routing rules applied by code after the LLM decided (set by the
    # Monitoring Agent; the LLM never produces it).
    orchestration_notes: SkipJsonSchema[list[str]] = Field(default_factory=list)
