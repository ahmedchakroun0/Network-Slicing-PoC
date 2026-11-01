"""Workflow of one telemetry record, as a LangGraph StateGraph.

    START -> ingest -+-(no violation)-> skip_healthy -> END
                     +-(violation)----> monitoring -> near_rt_ric? -> non_rt_ric? -> finalize -> END

The state (RecordState) is shared by the nodes of ONE record and lives only
for that record's run. Memory across records stays in Redis (working memory)
and the results in MongoDB; each node does its own small writes there.

- ingest        SLA analysis, base-station configuration and persistence
                (Redis); opens the record document (MongoDB) on a violation.
- skip_healthy  no violation: the slice history keeps the record, no agent.
- monitoring    the Monitoring Agent's decision (LLM + deterministic routing).
- near_rt_ric / non_rt_ric
                one node per selected RIC agent, Near-RT first: the Non-RT RIC
                sees the Near-RT calls of this record in its working memory.
                The trusted context comes from the state, never from the LLM.
- finalize      closes the record document and counts it on the run.

One graph run per record gives one LangSmith trace per record (when tracing
is enabled), with every node and LLM call nested under it.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.monitoring_agent import MonitoringAgent
from src.memory.working_memory import RedisWorkingMemory
from src.repositories.mongo_repository import MongoRepository
from src.schema.monitoring import AgentName, MonitoringDecision, SLAAnalysis, TelemetryRecord
from src.schema.working_memory import RecordEvent

# Order in which the selected RIC agents run within one record.
AGENT_ORDER: tuple[AgentName, ...] = ("near_rt_ric", "non_rt_ric")


class RecordState(TypedDict, total=False):
    # Input
    record: TelemetryRecord
    seq: int
    # Set by deterministic code (ingest)
    sla_analysis: SLAAnalysis
    persistence: dict[str, int]
    slice_state: dict[str, Any]
    # Set by the Monitoring Agent
    decision: MonitoringDecision
    # One entry per invoked RIC agent: its typed result (None on failure)
    agent_results: Annotated[dict[str, Any], operator.or_]
    outcome: Literal["skipped", "done"]


def slice_state(record: TelemetryRecord) -> dict[str, Any]:
    """Configuration and resource context the RIC agents and governance get."""
    return {
        "slice_prb": record.slice_prb,
        "cell_prb": record.cell_prb,
        "slice_prb_share_pct": round(100 * record.slice_prb / record.cell_prb, 1),
        "prb_utilisation": record.prb_utilisation,
        "cell_spare_prb": record.cell_spare_prb,
        "scheduler": record.scheduler,
    }


def trusted_context(state: RecordState) -> dict[str, Any]:
    """Context the RIC agents and the governance gate depend on, built from
    the state (deterministic code) only."""
    record = state["record"]
    return {
        "target": record.cell_id,
        "slice_id": record.slice_id,
        "window_start_s": record.window_start_s,
        "sla_analysis": state["sla_analysis"].model_dump(mode="json"),
        "persistence": state["persistence"],
        "slice_state": state["slice_state"],
    }


def _status(result: Any) -> str | None:
    if result is None:
        return "error"
    if isinstance(result, dict):
        return result.get("status")
    return getattr(result, "status", None)


class RecordWorkflow:
    """Builds and runs the record graph of one dataset run."""

    def __init__(
        self,
        monitoring: MonitoringAgent,
        memory: RedisWorkingMemory | None,
        mongo_repository: MongoRepository | None = None,
        run_id: str | None = None,
    ) -> None:
        self.monitoring = monitoring
        self.memory = memory
        self.store = mongo_repository if run_id is not None else None
        self.run_id = run_id
        self.graph = self._build()

    def _build(self):
        graph = StateGraph(RecordState)
        graph.add_node("ingest", self.ingest)
        graph.add_node("skip_healthy", self.skip_healthy)
        graph.add_node("monitoring", self.monitoring_decision)
        for agent in AGENT_ORDER:
            graph.add_node(agent, self._ric_node(agent))
        graph.add_node("finalize", self.finalize)

        graph.add_edge(START, "ingest")
        graph.add_conditional_edges(
            "ingest",
            lambda s: "monitoring" if s["sla_analysis"].violations else "skip_healthy",
            ["monitoring", "skip_healthy"],
        )
        graph.add_edge("skip_healthy", END)
        for position, node in enumerate(("monitoring", *AGENT_ORDER)):
            later = list(AGENT_ORDER[position:])
            graph.add_conditional_edges(node, self._next_agent(later), [*later, "finalize"])
        graph.add_edge("finalize", END)
        return graph.compile(name="record_workflow")

    @staticmethod
    def _next_agent(candidates: list[str]):
        """Route to the next selected agent among candidates, else finalize."""

        def route(state: RecordState) -> str:
            decision = state["decision"]
            selected = {i.agent for i in decision.selected_agents} if decision.action_required else set()
            return next((a for a in candidates if a in selected), "finalize")

        return route

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def ingest(self, state: RecordState) -> dict[str, Any]:
        record = state["record"]
        analysis = MonitoringAgent.analyze_sla(record)
        persistence: dict[str, int] = {}
        if self.memory is not None:
            # Base-station configuration (a new run of the cell clears its
            # memory) and per-slice violation persistence.
            await self.memory.observe(record)
            persistence = await self.memory.update_persistence(record, analysis.violations)
        if analysis.violations and self.store is not None:
            self.store.insert_record(
                run_id=self.run_id, record_id=record.record_id, seq=state.get("seq", 0),
                telemetry=record.model_dump(mode="json"),
                sla=analysis.model_dump(mode="json"), persistence=persistence,
            )
        return {"sla_analysis": analysis, "persistence": persistence,
                "slice_state": slice_state(record)}

    async def skip_healthy(self, state: RecordState) -> dict[str, Any]:
        # The Monitoring Agent is not invoked; the slice history keeps the
        # record (recovery stays visible) and the run counts it.
        record = state["record"]
        if self.memory is not None:
            await self.memory.record_decision(
                record.cell_id, RecordEvent.of(record, state["sla_analysis"]), cell_level=False
            )
        if self.store is not None:
            self.store.count(run_id=self.run_id, counters={"rows_without_violation": 1})
        return {"outcome": "skipped"}

    async def monitoring_decision(self, state: RecordState) -> dict[str, Any]:
        decision = await self.monitoring.decide(
            state["record"], state["persistence"], state["sla_analysis"]
        )
        return {"decision": decision}

    def _ric_node(self, agent: AgentName):
        async def invoke(state: RecordState) -> dict[str, Any]:
            invocation = next(i for i in state["decision"].selected_agents if i.agent == agent)
            result = await self.monitoring.invoke_agent(
                invocation, state["record"].record_id, trusted_context(state)
            )
            if self.store is not None:
                self.store.count(run_id=self.run_id,
                                 counters={f"ric_results.{agent}.{_status(result)}": 1})
            return {"agent_results": {agent: result}}

        invoke.__name__ = agent
        return invoke

    async def finalize(self, state: RecordState) -> dict[str, Any]:
        decision = state["decision"]
        if self.store is not None:
            self.store.finish_record(run_id=self.run_id, record_id=state["record"].record_id)
            counters = {"rows_processed": 1, "action_required": int(decision.action_required)}
            for invocation in decision.selected_agents:
                counters[f"agent_invocations.{invocation.agent}"] = 1
            self.store.count(run_id=self.run_id, counters=counters)
        return {"outcome": "done"}

    # ------------------------------------------------------------------

    async def run(self, record: TelemetryRecord, seq: int = 0) -> RecordState:
        """Run the graph for one record: one trace named after the record."""
        return await self.graph.ainvoke(
            {"record": record, "seq": seq, "agent_results": {}},
            config={
                "run_name": f"record {record.record_id}",
                "tags": [record.cell_id, record.slice_id],
                "metadata": {
                    "run_id": self.run_id, "record_id": record.record_id, "seq": seq,
                    "cell_id": record.cell_id, "slice_id": record.slice_id,
                },
            },
        )

    def draw_mermaid(self) -> str:
        """The workflow diagram, generated from the graph itself."""
        return self.graph.get_graph().draw_mermaid()
