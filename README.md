# Agentic AI for 6G Networks — Agentic O-RAN Management PoC

## Overview

This project is a research Proof of Concept (PoC) for **Agentic AI in 6G Networks**, with a particular focus on autonomous and governed network management.

The objective is **not to implement a complete 6G or O-RAN network**. Instead, the PoC focuses on the design and implementation of an **agentic system capable of observing network conditions, reasoning about them, selecting specialized agents, proposing actions, and validating those actions through a dedicated governance layer**.

The network dataset used by the PoC represents the type of telemetry that would normally be produced by a monitoring system. Individual dataset records are therefore treated as **telemetry inputs** to the agentic architecture.

The core research question is:

> **How can a multi-agent AI system autonomously reason about network conditions, delegate decisions to specialized RIC agents, and ensure that proposed network actions remain subject to explicit trust, policy, SLA, and safety constraints?**

The PoC implements four cooperating agents:

1. **Monitoring Agent — Orchestrator**
2. **Near-RT RIC Agent**
3. **Non-RT RIC Agent**
4. **RAN Trust & Governance Agent**

The architecture intentionally separates **reasoning**, **specialized decision making**, and **deterministic governance**.

---

# 1. Research Objective

The primary objective is to demonstrate an **agentic decision-making architecture for autonomous RAN management**.

The system is designed around the following conceptual pipeline:

```text
Network Telemetry
       │
       ▼
Monitoring Agent
       │
       │  Situation understanding
       │  SLA interpretation
       │  Agent selection
       ▼
RIC Agent
       │
       │  Action reasoning
       │  Action proposal
       ▼
Trust & Governance Agent
       │
       │  Validation
       │  Policy
       │  Trust
       │  SLA
       │  Safety
       ▼
Approved / Rejected
```

The PoC deliberately stops before real network execution.

A proposed action is therefore considered an **intent or control decision**, rather than an actual modification of a RAN.

This allows the research prototype to concentrate on the agentic mechanisms themselves without requiring a real O-RAN deployment.

---

# 2. Architectural Principles

The architecture follows several fundamental principles.

## 2.1 Centralized orchestration

The **Monitoring Agent is the entry point of the system**.

It is the only component that receives the incoming telemetry records.

It is responsible for determining:

- what is happening;
- whether the network situation requires intervention;
- how severe the situation is;
- which specialized agent should handle it;
- whether Near-RT RIC, Non-RT RIC, both, or neither should be invoked.

The Monitoring Agent does **not** directly execute network actions.

---

## 2.2 Specialized agents

The system separates RAN reasoning according to the temporal and operational responsibilities traditionally associated with Near-RT and Non-RT RIC functions.

The **Near-RT RIC Agent** handles short-term and relatively rapid decisions.

The **Non-RT RIC Agent** handles higher-level and longer-term decisions.

This separation is important because the goal is not to create one large general-purpose network agent.

Instead, the architecture demonstrates **specialized autonomous agents coordinated by an orchestrator**.

---

## 2.3 Governance as an interception layer

RIC agents are allowed to reason about potential actions, but they cannot simply declare an action executable.

Every protected action must pass through the **RAN Trust & Governance Agent**.

Therefore:

```text
RIC Agent
   │
   │ Proposed Action
   ▼
Trust & Governance
   │
   ├── Schema validation
   ├── Policy validation
   ├── Trust/Security validation
   ├── SLA validation
   └── Resource/Safety validation
   │
   ▼
Approved / Rejected
```

The governance layer acts as a **control boundary between AI-generated decisions and potential network operations**.

This is one of the central principles of the PoC.

---

# 3. The Four-Agent System

## 3.1 Monitoring Agent

The Monitoring Agent is the **main orchestrator**.

It receives one telemetry record at a time and transforms that raw observation into a structured decision.

Its responsibilities are:

- interpreting incoming telemetry;
- understanding the current network/slice situation;
- evaluating SLA conditions;
- identifying relevant degradation;
- considering recent observations through working memory;
- determining severity;
- deciding whether intervention is necessary;
- selecting the appropriate RIC agent;
- explaining the decision at a high level.

The Monitoring Agent can produce four routing outcomes:

```text
No intervention
Near-RT RIC
Non-RT RIC
Both RIC agents
```

The important distinction is that the Monitoring Agent **does not implement the RAN optimization itself**.

It decides **who should reason about the problem**.

---

## 3.2 Near-RT RIC Agent

The Near-RT RIC Agent represents an autonomous agent responsible for relatively fast RAN decisions.

Its conceptual responsibilities include:

- short-term resource optimization;
- rapid mitigation of degradation;
- radio/resource adjustments;
- local optimization.

In the PoC, these are represented as **agentic action proposals** rather than real RAN operations.

For example, the agent could reason that a slice experiencing high latency should receive additional resources and generate a structured proposal such as:

```text
Target:
    ran_slice

Action:
    resource_reallocation

Parameters:
    resource = PRB
    direction = increase

Expected effect:
    reduce_latency
```

The important research aspect is not the physical PRB modification.

The important aspect is that an autonomous agent:

1. receives a structured objective;
2. reasons about a possible intervention;
3. generates a machine-readable action;
4. submits that action to governance.

---

## 3.3 Non-RT RIC Agent

The Non-RT RIC Agent represents a higher-level optimization agent.

Its responsibilities include:

- policy-level optimization;
- longer-term resource planning;
- persistent degradation analysis;
- higher-level RAN configuration decisions.

The Non-RT RIC Agent receives a structured request from the Monitoring Agent.

It then reasons about the problem and generates a proposed action.

As with the Near-RT RIC Agent, the proposal is **not directly executable**.

It must pass through the Trust & Governance Agent.

---

## 3.4 RAN Trust & Governance Agent

The RAN Trust & Governance Agent provides the safety and governance boundary for the system.

It receives:

- the proposed action;
- the relevant context;
- the originating RIC agent;
- applicable policy information;
- SLA information;
- resource/safety constraints.

It does **not** independently consume the telemetry dataset.

Its responsibility is to answer:

> **Is this proposed action allowed to proceed under the defined governance constraints?**

The validation pipeline is deterministic wherever possible:

```text
Proposed Action
      │
      ▼
Action Schema Validation
      │
      ▼
Policy / Trust Validation
      │
      ▼
SLA Validation
      │
      ▼
Resource / Safety Validation
      │
      ▼
Approved / Rejected
```

A rejected action must never be considered executable.

---

# 4. Agent Interaction Model

The complete agentic interaction is:

```text
Telemetry Record
       │
       ▼
Monitoring Agent
       │
       ├──────────────► No Intervention
       │
       ├──────────────► Near-RT RIC Agent
       │                       │
       │                       ▼
       │                Proposed Action
       │                       │
       │                       ▼
       │              Trust & Governance
       │                       │
       │                       ▼
       │                Approved / Rejected
       │
       ├──────────────► Non-RT RIC Agent
       │                       │
       │                       ▼
       │                Proposed Action
       │                       │
       │                       ▼
       │              Trust & Governance
       │                       │
       │                       ▼
       │                Approved / Rejected
       │
       └──────────────► Both
```

A fundamental architectural constraint is:

```text
Monitoring Agent
       │
       ├── Near-RT RIC
       │       └── Trust & Governance
       │
       └── Non-RT RIC
               └── Trust & Governance
```

There is no:

```text
Monitoring Agent → Trust & Governance
```

and no:

```text
RIC Agent → Dataset
```

The RIC agents operate on the **structured context supplied by the Monitoring Agent**, not on the original telemetry stream.

This preserves the separation of responsibilities defined by the architecture.

---

# 5. Telemetry as the Agentic Input

The telemetry comes from the **ColO-RAN** dataset (real srsRAN stack on the Colosseum emulator: 7 base stations, 3 slices each, 3 schedulers x 28 PRB splits). `scripts/prepare_coloran.py` aggregates the per-UE KPM reports (every 250 ms) into one `TelemetryRecord` per **base station, slice and 1 s window**, in native units. The CSV columns are the record's field names, so a row validates as-is (`src/schema/monitoring.py`):

| Group | Fields |
|---|---|
| Identity | `record_id`, `cell_id`, `slice_id` (eMBB / MTC / URLLC), `run` (source configuration), `window_start_s`, `window_s` |
| Configuration | `slice_prb` (of `cell_prb` = 50), `scheduler` |
| Measurements | `active_ues`, `dl_throughput_mbps`, `dl_throughput_per_ue_mbps`, `dl_buffer_bytes`, `queueing_delay_ms`, `prb_utilisation`, `cell_spare_prb`, `dl_cqi`, `dl_mcs`, `ul_sinr_db` |

`scripts/explore_coloran.py` documents why these fields were chosen (columns that never change, redundant counters, the start-up window). Each record is processed independently and sequentially.

---

# 6. Deterministic SLA Evaluation

Numerical SLA comparisons are not delegated to the LLM. The records carry measurements only; the SLA profile of each slice type lives in `src/schema/sla.py` and is evaluated at runtime by `MonitoringAgent.analyze_sla`:

| Slice | throughput per UE (min) | queueing delay (max) |
|---|---|---|
| eMBB | 3.6 Mbps (90 % of the 4 Mbps CBR load) | 100 ms |
| MTC | 0.015 Mbps | 100 ms |
| URLLC | 0.005 Mbps | 10 ms |

The result (`SLAAnalysis`) lists every KPI with its measured value, budget, unit and bound, plus the violated and unavailable KPIs. The LLM reasons over it and never recomputes it. PRB utilisation and spare PRBs are context, not SLA terms.

---

# 7. Agentic Reasoning

The system should not be implemented as a collection of fixed `if/else` rules.

The deterministic components establish **facts and constraints**.

The LLM provides the **reasoning and orchestration layer**.

For example, deterministic tools may determine:

```text
Latency violation = true
Throughput violation = true
Packet loss violation = false
```

The Monitoring Agent then reasons about:

```text
What is the current situation?

How severe is it?

Does it require intervention?

Which specialized agent should handle it?

Should both short-term and long-term optimization be considered?

Why?
```

The output is converted into a strongly typed decision.

This allows the system to demonstrate genuine agentic behavior while retaining deterministic control over safety-critical logic.

---

# 8. Structured Agent Communication

Agents communicate using strongly typed Pydantic models rather than unstructured natural-language messages.

Important conceptual objects include:

```text
TelemetryRecord
MonitoringDecision
AgentRequest
ProposedAction
GovernanceDecision
```

For example, the Monitoring Agent may produce:

```json
{
  "situation": "critical",
  "sla_violations": ["latency", "throughput"],
  "severity": "high",
  "selected_agents": ["near_rt_ric", "non_rt_ric"],
  "action_required": true,
  "reason": "The slice is experiencing significant SLA degradation requiring immediate mitigation and longer-term optimization."
}
```

The exact schemas can evolve as the research progresses, but the principle remains:

> **Agent-to-agent communication should be machine-readable, validated, and explicit.**

This makes the system easier to test and prevents free-form LLM output from becoming an uncontrolled interface between agents.

---

# 9. Proposed Actions

RIC agents do not return arbitrary textual recommendations.

They generate structured action proposals.

Conceptually:

```json
{
  "target": "ran_slice",
  "action_type": "resource_reallocation",
  "parameters": {
    "resource": "PRB",
    "direction": "increase"
  },
  "expected_effect": "reduce_latency",
  "priority": "high"
}
```

An action contains enough information for the governance layer to understand:

- what is being targeted;
- what operation is being proposed;
- which parameters are involved;
- what effect is expected;
- how important the action is.

The action vocabulary remains intentionally abstract.

For example:

```text
resource_reallocation
scheduler_optimization
handover_optimization
power_adjustment
```

These represent **research-level control intents**, not implementations of real O-RAN control mechanisms.

---

# 10. Trust and Governance

The Trust & Governance Agent converts an AI-generated proposal into a governed decision.

A governance result can conceptually contain:

```json
{
  "approved": true,
  "checks": {
    "schema": true,
    "policy": true,
    "trust": true,
    "sla": true,
    "resource_safety": true
  },
  "reason": "Action satisfies the defined governance constraints."
}
```

A rejected action could produce:

```json
{
  "approved": false,
  "checks": {
    "schema": true,
    "policy": false,
    "trust": true,
    "sla": true,
    "resource_safety": true
  },
  "reason": "The proposed action violates the configured policy."
}
```

The important property is:

```text
LLM proposes
     ↓
Governance validates
     ↓
System decides whether it is allowed
```

The LLM cannot bypass governance.

---

# 11. Deterministic Guardrails

The governance layer should use deterministic validation whenever possible.

The conceptual guardrail order is:

```text
1. Action Schema Validation
          ↓
2. Policy / Trust Validation
          ↓
3. SLA Validation
          ↓
4. Resource / Safety Validation
          ↓
5. Final Governance Decision
```

This allows the system to distinguish between:

- what the LLM **believes should happen**;
- what the system **allows to happen**.

That distinction is particularly important for an agentic networking architecture.

---

# 12. Working Memory

The PoC uses **working memory only**.

The purpose of working memory is to provide recent context to the agents.

It can contain:

```text
Recent telemetry observations
Recent SLA violations
Recent events
Previous decisions
Previously invoked agents
Recent proposed actions
Recent governance outcomes
```

For example:

```text
Observation 1
    latency degradation
    → Near-RT selected

Observation 2
    latency degradation persists
    → Near-RT + Non-RT selected

Observation 3
    governance rejected previous action
    → Monitoring Agent considers alternative
```

This enables the system to reason about **evolving network conditions**, rather than treating every telemetry record as completely independent.

The memory remains deliberately simple.

The PoC does not require:

- vector databases;
- semantic memory;
- knowledge graphs;
- long-term memory;
- complex distributed memory systems.

The memory interface should nevertheless be designed so that persistent memory can be introduced later without redesigning the agents.

---

# 13. LLM Abstraction

The agents communicate with models through **LiteLLM**.

The agents should not be tightly coupled to a specific model provider.

Conceptually:

```text
Agent
  │
  ▼
LLM Gateway
  │
  ├── OpenAI
  ├── Azure OpenAI
  ├── vLLM
  └── Other compatible models
```

The model is therefore an implementation detail of the agentic system.

This is important for research experimentation because different models can later be evaluated under the same agent architecture.

The PoC can therefore study questions such as:

- Does model selection affect agent routing?
- Does model capability affect action quality?
- How often does governance reject unsafe proposals?
- How consistent are decisions across models?
- How does working memory affect sequential decisions?

---

# 14. Agent State

Each agent should operate using structured state rather than relying exclusively on conversational text.

Conceptually, state can include:

```text
Current invocation
Current context
Relevant SLA analysis
Working memory
Agent decision
Proposed action
Governance outcome
```

The Monitoring Agent's state represents the orchestration context.

A RIC Agent's state represents the specialized decision context.

The Trust & Governance Agent's state represents the action-validation context.

This separation prevents unrelated information from being passed indiscriminately between agents.

---

# 15. Separation of Responsibilities

The architecture intentionally separates the system into four conceptual layers.

### Observation

```text
Telemetry
```

### Reasoning

```text
Monitoring Agent
RIC Agents
```

### Governance

```text
Trust & Governance Agent
```

### Execution

```text
Not implemented in the PoC
```

Therefore:

```text
Observation
    ↓
Reasoning
    ↓
Governance
    ↓
[Future Execution Layer]
```

This separation makes the PoC suitable as a foundation for later integration with real O-RAN infrastructure.

---

# 16. Execution Model

The PoC processes telemetry sequentially.

For each record:

```text
1. Read telemetry record
        ↓
2. Convert it into a validated TelemetryRecord
        ↓
3. Analyze deterministic SLA conditions
        ↓
4. Provide structured context to Monitoring Agent
        ↓
5. Monitoring Agent reasons about the situation
        ↓
6. Monitoring Agent selects zero, one, or two RIC agents
        ↓
7. Selected RIC agent receives a structured request
        ↓
8. RIC agent reasons about a possible action
        ↓
9. RIC agent generates ProposedAction
        ↓
10. ProposedAction enters governance
        ↓
11. Governance validates the action
        ↓
12. Action is approved or rejected
        ↓
13. Result is recorded
        ↓
14. Working memory is updated
```

The process is then repeated for the next observation.

This provides a simple but complete agentic loop.

Since v3.3 the workflow of one record is a LangGraph `StateGraph`
([agents/record_workflow.py](agents/record_workflow.py)). Its state is shared
by the nodes of that record only; Redis keeps the memory across records and
MongoDB the results. The diagram below is generated by
`RecordWorkflow.draw_mermaid()`:

```mermaid
graph TD;
	__start__([start]) --> ingest;
	ingest -.->|violation| monitoring;
	ingest -.->|no violation| skip_healthy;
	monitoring -.-> near_rt_ric;
	monitoring -.-> non_rt_ric;
	monitoring -.-> finalize;
	near_rt_ric -.-> non_rt_ric;
	near_rt_ric -.-> finalize;
	non_rt_ric -.-> finalize;
	finalize --> __end__([end]);
	skip_healthy --> __end__;
```

| Node | Does | Writes |
|---|---|---|
| `ingest` | SLA analysis, base-station configuration, persistence | Redis meta + persistence; MongoDB record (on a violation) |
| `skip_healthy` | no violation: no LLM call | Redis slice history; run counter |
| `monitoring` | LLM decision + deterministic routing rules | MongoDB `monitoring`; Redis cell/slice history |
| `near_rt_ric`, `non_rt_ric` | the selected RIC agents, Near-RT first, with the trusted context built from the state | MongoDB `agents.<agent>`; Redis actions |
| `finalize` | closes the record | MongoDB status + run counters |

One graph run per record is one LangSmith trace (`record <record_id>`, with
the run, record, cell and slice as metadata) when `LANGSMITH_TRACING=true`.

---

# 17. Example End-to-End Scenario

Consider a telemetry record indicating that:

```text
Latency:
    above budget

Throughput:
    below required rate

Packet loss:
    within budget
```

The deterministic SLA component returns:

```text
Latency violation = true
Throughput violation = true
Packet loss violation = false
```

The Monitoring Agent receives these facts and reasons that the slice is experiencing significant degradation.

It may produce:

```text
Situation:
    SLA degradation

Severity:
    High

Action required:
    True

Selected agents:
    Near-RT RIC
    Non-RT RIC
```

The Near-RT RIC Agent may propose an immediate resource adjustment.

The Non-RT RIC Agent may propose a longer-term optimization.

Each proposal is independently sent through governance.

The final result could therefore be:

```text
Monitoring
    ├── Near-RT RIC
    │       └── Action A
    │              └── Governance → APPROVED
    │
    └── Non-RT RIC
            └── Action B
                   └── Governance → REJECTED
```

The system therefore demonstrates that **agent selection, action generation, and action authorization are independent decisions**.

---

# 18. Observability

The PoC should provide a clear execution trace.

A typical execution should be understandable from logs such as:

```text
Record: 00042

Monitoring Agent
    Situation: SLA degradation
    Violations: latency, throughput
    Severity: high
    Selected agents:
        - near_rt_ric
        - non_rt_ric

Near-RT RIC
    Proposed action: resource_reallocation

Trust & Governance
    Result: APPROVED

Non-RT RIC
    Proposed action: policy_adjustment

Trust & Governance
    Result: REJECTED
```

The logs should expose:

- agent invocation;
- structured inputs;
- structured outputs;
- selected agents;
- proposed actions;
- governance decisions;
- execution outcomes.

They should **not expose private chain-of-thought**.

Only concise explanations and structured decision information should be logged.

---

# 19. Research-Oriented Evaluation

The architecture makes it possible to evaluate the agentic system independently from a real RAN.

Potential evaluation dimensions include:

### Monitoring decisions

Measure:

- correctness of agent selection;
- correctness of intervention decisions;
- severity classification;
- consistency of decisions.

### RIC decisions

Measure:

- action validity;
- action relevance;
- alignment with the Monitoring Agent request;
- structured-output validity.

### Governance

Measure:

- percentage of actions approved;
- percentage rejected;
- reasons for rejection;
- policy violations;
- invalid actions detected.

### End-to-end behavior

Measure:

```text
Telemetry
   ↓
Monitoring
   ↓
Agent Selection
   ↓
Action Proposal
   ↓
Governance
   ↓
Final Decision
```

The dataset therefore becomes an evaluation input for the **agentic architecture**, rather than an attempt to benchmark actual 6G network performance.

---

# 20. What the PoC Does Not Implement

The current research prototype intentionally does not implement real network control.

It does not require:

- real O-RAN interfaces;
- E2;
- A1;
- O1;
- real xApps;
- real RAN deployment;
- Kubernetes;
- ns-3;
- FlexRIC;
- real-time radio control;
- actual PRB allocation;
- actual scheduler modification;
- actual handover control.

Instead, these capabilities are represented through clean abstractions and structured action proposals.

This allows the research to focus on the **agentic intelligence and governance architecture** first.

Real network interfaces can be introduced later behind the same action abstraction.

---

# 21. Future Integration

The PoC is intentionally designed so that the final stage can later evolve from:

```text
Proposed Action
      ↓
Governance
      ↓
Simulation / Result
```

toward:

```text
Proposed Action
      ↓
Governance
      ↓
O-RAN Control Interface
      ↓
Real Network
      ↓
Telemetry
      ↓
Monitoring Agent
```

This creates a closed autonomous control loop:

```text
Observe
   ↓
Reason
   ↓
Delegate
   ↓
Propose
   ↓
Govern
   ↓
Execute
   ↓
Observe Again
```

The current PoC implements the first five stages without requiring a physical or simulated RAN.

---

# 22. Core Design Principle

The central principle of this research prototype is:

> **AI agents can reason about network conditions and propose autonomous actions, but network actions must remain subject to explicit deterministic governance.**

The resulting architecture can be summarized as:

```text
                    ┌─────────────────────┐
                    │   Network Telemetry │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Monitoring Agent   │
                    │    Orchestrator     │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        ┌──────────────────┐       ┌──────────────────┐
        │ Near-RT RIC      │       │ Non-RT RIC       │
        │ Agent            │       │ Agent            │
        └────────┬─────────┘       └────────┬─────────┘
                 │                          │
                 └────────────┬─────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ RAN Trust & Governance  │
                 │         Agent           │
                 └────────────┬────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 APPROVED             REJECTED
                    │
                    ▼
             Future RAN Execution
```

The PoC therefore provides a **minimal but research-oriented foundation for Agentic AI in autonomous 6G/O-RAN management**.

Its contribution at this stage is the **agentic architecture, inter-agent coordination, structured decision making, working memory, action generation, and trust-aware governance**, rather than implementation of the underlying 6G network functions. 17. Running the PoC

The runtime lifecycle is intentionally simple.

17.1 Start Redis

Redis must be running before starting the agentic system because it provides the working-memory backend.

On a Linux system using the Redis service:

sudo systemctl start redis-server

Verify that Redis is available:

redis-cli ping

Expected result:

PONG

If Redis is already configured to start automatically, only the verification step is necessary.

17.2 Start MongoDB

MongoDB provides persistent storage for agent execution information and results.

Start the MongoDB service:

sudo systemctl start mongod

Verify that MongoDB is running:

sudo systemctl status mongod

A successful installation should show the service as:

active (running)

MongoDB can also be checked directly with:

mongosh

and then:

db.runCommand({ ping: 1 })

The expected response contains:

ok: 1

18. Run Tests

The test suite should be executed before running the complete dataset.

For a Poetry-based environment:

poetry run pytest

or, if the project is already inside the Poetry environment:

pytest

The tests should cover at least:

SLA calculations
Pydantic schemas
Working memory
Redis persistence
MongoDB persistence
Governance rules
Agent routing
Invalid actions
Approved actions
Rejected actions
Agent-to-agent communication

The most important integration test should validate the complete workflow:

Telemetry
↓
Monitoring Agent
↓
Agent Selection
↓
RIC Agent
↓
Proposed Action
↓
Trust & Governance
↓
Approved / Rejected
↓
Persistence

LLM calls should be mocked during unit and integration tests whenever possible.

This ensures that tests validate the agentic system rather than network availability or model-provider availability.

19. Run the Dataset

After Redis and MongoDB are available and the tests pass, the dataset can be used as the telemetry source.

A small number of records should be tested first:

poetry run python scripts/run_dataset.py --max-rows 10

The exact argument can depend on the implementation.

The execution should:

Load the telemetry dataset.

Process records sequentially.

Send each record to the Monitoring Agent.

Let the Monitoring Agent determine the required intervention.

Invoke the selected RIC agents.

Generate structured proposed actions.

Pass protected actions through Trust & Governance.

Record tool executions and decisions.

Update Redis working memory.

Persist relevant execution information in MongoDB.

Once the small-scale execution is validated, the complete dataset can be processed.

20. Recommended Run Sequence

The complete development workflow is:

Start Redis
↓
Verify Redis
↓
Start MongoDB
↓
Verify MongoDB
↓
Run Tests
↓
Run a Small Dataset Sample
↓
Inspect Agent Trace
↓
Inspect Redis Working Memory
↓
Inspect MongoDB Execution Records
↓
Run Full Dataset

In command form:

sudo systemctl start redis-server
redis-cli ping

sudo systemctl start mongod
sudo systemctl status mongod

poetry run pytest

poetry run python scripts/run_dataset.py --max-rows 10

After validating the first records:

poetry run python scripts/run_dataset.py


# v2 corrections

See `CHANGES.md` for the list of fixes (role binding, Non-RT persistence gate, per-(cell, slice) persistence, optional `Cell ID`, bounded working memory, audit fix, prompt/test/path fixes) and the new end-to-end test; section v3 lists the fixes from the live-LLM runs (default-deny safety stage, schema alignment of the new tools, invocation limits, merged duplicate agent selections). Run `poetry run pytest` (40 tests) before `scripts/run_dataset.py`; add `--debug` for verbose LangChain tracing.


# Using the ColO-RAN dataset (real-stack wireless traces)

The original `data/prepared_network_slicing_dataset.csv` is synthetic (Botez et al., Electronics 2025).
The PoC can also run on **ColO-RAN** (Polese et al., IEEE TMC 2023): per-UE traces collected with an
srsRAN-based stack on the Colosseum emulator — 7 base stations, 3 slices (eMBB, MTC, URLLC),
3 schedulers x 28 RBG splits. No agent code changes: the preprocessing writes the same CSV columns.

```bash
# 1. download (sparse clone, ~750 MB for one experiment run; GPL-3.0 data)
bash scripts/download_coloran.sh                       # -> data/raw/coloran (git-ignored)

# 2. convert to Monitoring-Agent telemetry (~5 min, 724,086 records)
poetry run python scripts/prepare_coloran.py           # -> data/coloran_prepared_dataset.csv (+ .summary.json)
#    options: --runs 50 (random subset of cell runs), --window 1.0 (s per record), --experiments exp1 exp2

# 3a. dry run without model / Redis / MongoDB (scripted LLM)
poetry run python scripts/demo_run_scripted.py 60

# 3b. real run (LLM via LiteLLM, Redis and MongoDB running)
poetry run python scripts/run_dataset.py data/coloran_prepared_dataset.csv --max-rows 300
```

What one record is: one slice of one base station over a 1-s window.

| CSV column (TelemetryRecord) | ColO-RAN source |
|---|---|
| Slice Available Transfer Rate (Gbps) | sum of the slice UEs' DL rate (`tx_brate downlink`) |
| Slice Latency (ns) | mean queueing-delay estimate `dl_buffer * 8 / DL rate` (capped at 2 s) |
| Slice Jitter (ns) | standard deviation of that estimate in the window |
| Slice Packet Loss | mean DL transmission-error ratio (`tx_errors downlink (%)` / 100) |
| Slice Type / Use Case Type | eMBB, MTC, URLLC (slice 0/1/2) |
| Cell ID | base station (`bs1`..`bs7`) -> used as the governance target |
| budgets | fixed a priori per slice (see `BUDGETS` in `scripts/prepare_coloran.py`) |
| Violation labels | computed with `MonitoringAgent.analyze_sla` |

Records are ordered as a telemetry stream: per (scheduler, split, BS) run, by time, the three slices
interleaved. Each run starts when all three slices carry traffic (UE attach takes ~50 s).
Extra columns (`Scheduler`, `RBG Split`, `Run`, `Window Start (s)`) are ignored by the agents.


# Working memory and results store (v3.3)

**Redis is the agents' working memory**: network context for the next decision, keyed by base station and slice (`src/memory/working_memory.py`). Lists are capped with `LPUSH` + `LTRIM`, and every read or write is one pipeline.

| Key | Type | Content |
|---|---|---|
| `wm:cell:{cell}:meta` | HASH | configuration in force: run, scheduler, `prb:<slice>`, `util:<slice>` |
| `wm:cell:{cell}:records` / `wm:slice:{cell}:{slice}:records` | LIST | last records (KPIs, violations, severity, agents selected) |
| `wm:cell:{cell}:actions` / `wm:slice:{cell}:{slice}:actions` | LIST | last tool calls: tool, parameters, approved, executed, rejecting stage and deterministic reason |
| `wm:slice:{cell}:{slice}:persistence` | HASH | consecutive violated records per SLA KPI (Non-RT governance gate) |
| `wm:cells` | SET | cells seen (used by `reset`) |

Only facts are stored, never LLM rationales. A record from a new configuration of a base station clears that base station's memory.

**MongoDB is the results store** (`src/repositories/mongo_repository.py`): `runs` has one small document per run (configuration and counters), and `records` has one document per telemetry record (telemetry, SLA analysis, persistence, the Monitoring decision, and each RIC agent's tool calls with their governance outcome). Each agent step adds to the record with a targeted `$set` / `$push`; documents are never re-written whole.
