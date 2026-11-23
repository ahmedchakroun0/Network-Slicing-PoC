"""Modelled effects on top of a recorded trace row.

ColO-RAN has no data for transmit power, handover, QoS priority, link
adaptation, or for demand surges, interference and outages. Their effect is
modelled here, as simple and explainable rules, and applied to the trace row
of the configuration in force. With neutral controls and no event, the row is
returned unchanged (transparent replay).

Radio: an SINR shift moves the CQI by about 1 per 2 dB and the MCS by about
2 per CQI step. Capacity scales with the spectral efficiency of the CQI
(3GPP TS 38.214 Table 5.2.2.1-2).

Link adaptation: the trace already reflects the testbed's link adaptation
(BLER target about 10 %). An MCS offset, a different BLER target or a link
degradation the outer loop has not caught up with creates a mismatch: a higher
effective MCS gains about 6 % rate per step but multiplies the BLER by 1.8.

HARQ: each retransmission adds one HARQ round trip (8 ms); with BLER p a block
needs p / (1 - p) retransmissions on average. Only the part above the testbed's
own BLER is added, so the neutral case adds nothing.

Queue: the capacity of the slice's allocation is the trace rate divided by its
PRB utilisation, times the radio factor. Demand above capacity accumulates as an
extra backlog, which adds to the delay until it is served. Only BACKLOG_KEEP of
the unserved backlog is carried to the next window; the rest is lost (RLC buffer
overflow, TCP backing off), so a saturated slice recovers after an event ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Spectral efficiency (bit/s/Hz) per CQI 0..15, TS 38.214 Table 5.2.2.1-2.
CQI_EFFICIENCY = np.array(
    [
        0.0,
        0.1523,
        0.2344,
        0.3770,
        0.6016,
        0.8770,
        1.1758,
        1.4766,
        1.9141,
        2.4063,
        2.7305,
        3.3223,
        3.9023,
        4.5234,
        5.1152,
        5.5547,
    ]
)
BASE_BLER = 0.1  # BLER target of the testbed's link adaptation
BLER_GROWTH = 1.8  # BLER multiplier per effective MCS step above the target
RATE_PER_MCS_STEP = 0.06  # rate gain per effective MCS step
MIN_UTILISATION = 0.05  # floor when inferring capacity from utilisation
HARQ_RTT_MS = 8.0  # LTE/NR FDD HARQ round trip
BACKLOG_KEEP = 0.5  # share of the unserved backlog carried to the next window
NEUTRAL_HANDOVER_DB = 3.0
NEUTRAL_PRIORITY = 5


def efficiency(cqi: float) -> float:
    return float(np.interp(min(max(cqi, 1.0), 15.0), np.arange(16), CQI_EFFICIENCY))


def handover_sinr_db(margin_db: float) -> float:
    """SINR effect of the handover margin (single-cell model).

    A lower margin hands edge UEs to a better neighbour sooner, which raises the
    mean SINR slightly; below 1 dB the UEs ping-pong and lose more than they gain.
    """
    if margin_db >= NEUTRAL_HANDOVER_DB:
        return 0.0
    if margin_db >= 1.0:
        return 0.15 * (NEUTRAL_HANDOVER_DB - margin_db)
    return 0.3 - 1.5 * (1.0 - margin_db)


@dataclass(frozen=True)
class RadioEffect:
    cqi: float
    mcs: float
    rate_factor: float
    bler: float


def radio_effect(
    cqi: float,
    mcs: float,
    sinr_shift_db: float,
    mcs_offset: int,
    target_bler: float,
    la_lag_db: float,
) -> RadioEffect:
    """CQI, MCS and capacity factor after an SINR shift and a link-adaptation mismatch.

    la_lag_db: SINR loss the outer loop has not compensated yet (link degradation),
    so the scheduler keeps an MCS that is too high for the channel.
    """
    new_cqi = min(max(cqi + sinr_shift_db / 2.0, 1.0), 15.0)
    radio = efficiency(new_cqi) / efficiency(cqi)
    mismatch = mcs_offset + math.log(target_bler / BASE_BLER, BLER_GROWTH) + la_lag_db / 2.0
    bler = min(max(BASE_BLER * BLER_GROWTH**mismatch, 0.001), 0.6)
    la = (1 + RATE_PER_MCS_STEP * mismatch) * (1 - bler) / (1 - BASE_BLER)
    new_mcs = min(max(mcs + 2.0 * (new_cqi - cqi) + mcs_offset, 0.0), 28.0)
    return RadioEffect(cqi=new_cqi, mcs=new_mcs, rate_factor=max(radio * la, 0.0), bler=bler)


def harq_extra_delay_ms(bler: float) -> float:
    """Retransmission delay above the testbed's own (BASE_BLER) retransmissions."""
    return HARQ_RTT_MS * (bler / (1 - bler) - BASE_BLER / (1 - BASE_BLER))


@dataclass(frozen=True)
class QueueResult:
    throughput_mbps: float
    utilisation: float
    delay_ms: float
    buffer_bytes: float  # mean per UE, trace backlog + extra backlog
    backlog_bytes: float  # extra slice backlog carried to the next window
    dropped_bytes: float  # unserved backlog lost in this window


def queue_step(
    *,
    rate_mbps: float,
    utilisation: float,
    delay_ms: float | None,
    buffer_bytes: float,
    ues: int,
    slice_prb: int,
    window_s: float,
    rate_factor: float,
    surge: float,
    extra_prb: float,
    backlog_bytes: float,
    harq_extra_ms: float = 0.0,
    demand_mbps: float | None = None,
) -> QueueResult:
    """Serve one window of demand + carried backlog.

    demand_mbps defaults to the trace rate x surge; a surge passes the demand
    anchored at its start, so it does not grow when the slice gets more PRBs.
    """
    ues = max(ues, 1)
    cap_ms = window_s * 1e3
    prb_scale = (slice_prb + extra_prb) / slice_prb
    max_rate = rate_mbps / max(utilisation, MIN_UTILISATION) * rate_factor * prb_scale
    if demand_mbps is None:
        demand_mbps = rate_mbps * surge
    demand_bits = demand_mbps * window_s * 1e6 + backlog_bytes * 8
    served = min(demand_bits / (window_s * 1e6), max_rate)
    unserved = max(demand_bits - served * window_s * 1e6, 0.0) / 8
    util = min(served / max_rate * prb_scale, 1.0) if max_rate > 0 else 0.0
    if served <= 0:
        delay = cap_ms
    else:
        queue_ms = unserved * 8 / (served * 1e6) * 1e3
        delay = min(max((delay_ms or 0.0) + queue_ms + harq_extra_ms, 0.0), cap_ms)
    return QueueResult(
        throughput_mbps=served,
        utilisation=util,
        delay_ms=delay,
        buffer_bytes=buffer_bytes + unserved / ues,
        backlog_bytes=unserved * BACKLOG_KEEP,
        dropped_bytes=unserved * (1 - BACKLOG_KEEP),
    )
