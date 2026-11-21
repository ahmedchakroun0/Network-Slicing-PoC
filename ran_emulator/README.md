# RAN emulator (ColO-RAN traces)

A time-stepped emulator of one or more cells. Agents act on it, and the next KPIs react to their actions.
Each step is a 1 s window and emits one `TelemetryRecord` (`src/schema/monitoring.py`) per cell
and slice, in the same schema as the CSV replay. The Monitoring agent and the SLA analysis
work on it unchanged.

## How it works

| What | How | Source |
|---|---|---|
| PRB split (`adjust_prb_allocation`, slice policy) | **trace switching**: the cell continues, at the same window offset, in the recorded trace of the new split | measured (ColO-RAN) |
| Scheduler (`optimize_scheduler`) | trace switching; ColO-RAN uses one scheduler per cell, so the change applies to all slices | measured |
| Power, handover margin | SINR shift → CQI/MCS → capacity (3GPP CQI efficiency table) | modelled |
| Link adaptation (MCS offset, BLER target) | BLER mismatch → rate and HARQ retransmission delay | modelled |
| QoS priority | priority below 5 lets the slice borrow the cell's idle PRBs | modelled |
| Events: demand surge, interference, link degradation, cell outage | queue/backlog and radio overlay | modelled |

**Transparent replay:** with neutral controls and no event, the records are exactly the
recorded ColO-RAN rows. Everything modelled lives in `overlay.py`, with its rules in the module docstring.

**PRB snapping:** only 28 splits were recorded (eMBB in {6, 12, …, 42} PRBs, URLLC in {5, 11, …, 41},
MTC gets the rest), so a request is snapped to the closest recorded split that respects the slice
policies. Other slices are changed as little as possible: MTC gives up PRBs first, URLLC last. The
response reports the requested and the applied share.

## Run

```bash
poetry run python -m ran_emulator --scenario embb_prb_starvation          # lockstep, port 8100
poetry run python -m ran_emulator --scenario cell_interference --mode realtime --step-period 1
```

The full dataset (`data/coloran_prepared_dataset.csv`, 571 cell traces) takes about 15 s to load;
without it the tracked sample is used (fewer configurations, so more nearest-trace fallbacks).

```bash
curl -XPOST localhost:8100/sim/run -H 'content-type: application/json' -d '{"steps": 5}'
curl -XPUT  localhost:8100/cells/bs2/slices/eMBB/prb -H 'content-type: application/json' -d '{"prb_percentage": 72}'
curl -XPOST localhost:8100/sim/run -H 'content-type: application/json' -d '{"steps": 5}'
```

## API

| Endpoint | Body | Agent tool |
|---|---|---|
| `PUT /cells/{cell}/slices/{slice}/prb` | `prb_percentage` | `adjust_prb_allocation` |
| `PUT /cells/{cell}/scheduler` | `scheduler`, optional `slice_id` | `optimize_scheduler` |
| `PUT /cells/{cell}/handover` | `handover_margin_db` | `optimize_handover` |
| `PUT /cells/{cell}/power` | `power_offset_db` | `adjust_power_control` |
| `PUT /cells/{cell}/slices/{slice}/qos` | `priority_level`, `packet_delay_budget_ms`, `guaranteed_bitrate_mbps` | `adjust_qos_parameters` |
| `PUT /cells/{cell}/slices/{slice}/link-adaptation` | `mcs_offset`, `target_bler` | `adjust_link_adaptation` |
| `PUT /cells/{cell}/slices/{slice}/policy` | `min_prb_ratio_pct`, `max_prb_ratio_pct` | `update_slice_resource_policy` |
| `POST /sim/reset` | `scenario`, `seed` | |
| `POST /sim/run` | `steps` (lockstep only) | |
| `POST /sim/events` | an event (see the scenarios) | |
| `GET /state`, `GET /cells/{cell}/kpis?from_step&to_step` | | |
| `GET /sim/violations`, `GET /sim/actions` | | |
| `GET /telemetry/stream` | server-sent events, one per step | |

Each control returns `previous`, `current`, `effective_step` (always the next step) and a `note`.
A value outside the safety envelope (`src/schema/limits.py`, shared with Governance) is refused with
**422** and listed in `/sim/violations`: each entry is a call that Governance should have stopped.

`client.py` provides `EmulatorClient`, an async wrapper with one method per agent tool.

## Scenarios (`scenarios/`)

All start on bs2. Under the current SLA profiles, eMBB only meets its 3.6 Mbps per-UE budget with about 36 PRBs.

| Scenario | What breaks | Action that fixes it |
|---|---|---|
| `baseline_bs2` | nothing (36/3/11 PRBs) | none; the agents should stay quiet |
| `embb_prb_starvation` | eMBB throughput and delay (18 PRBs) | eMBB PRB share to about 72 % |
| `embb_demand_surge` | eMBB delay, demand × 1.2 | eMBB QoS priority 1 (PRBs alone are not enough) |
| `cell_interference` | eMBB throughput and delay, −6 dB | power offset +6 dB |
| `urllc_link_degradation` | URLLC delay (HARQ retransmissions) | URLLC MCS offset −4 or BLER target 0.01 |
| `cell_outage` | all slices, for 20 s | none; it recovers after the outage (tests escalation and patience) |
