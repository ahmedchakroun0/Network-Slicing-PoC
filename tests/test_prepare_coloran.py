"""Tests of scripts/prepare_coloran.py on a tiny synthetic tree that has the
exact ColO-RAN file layout and header (no download needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_coloran as prep

from agents.monitoring_agent import MonitoringAgent
from src.schema.monitoring import TelemetryRecord

# The real 36-column header, unnamed separator columns included.
HEADER = ("Timestamp,num_ues,IMSI,RNTI,,slicing_enabled,slice_id,slice_prb,power_multiplier,"
          "scheduling_policy,,dl_mcs,dl_n_samples,dl_buffer [bytes],tx_brate downlink [Mbps],"
          "tx_pkts downlink,tx_errors downlink (%),dl_cqi,,ul_mcs,ul_n_samples,ul_buffer [bytes],"
          "rx_brate uplink [Mbps],rx_pkts uplink,rx_errors uplink (%),ul_rssi,ul_sinr,phr,,"
          "sum_requested_prbs,sum_granted_prbs,,dl_pmi,dl_ri,ul_n,ul_turbo_iters")

# slice -> (slice_prb, DL Mbps per UE, backlog bytes, granted PRBs per UE report)
SPECS = {0: (30, 3.0, 50_000, 3750),   # eMBB: 2 x 4 x 3750 / (30 x 1000) = 100 % used
         1: (15, 0.06, 0, 150),        # MTC:  1200 / 15000 = 8 %
         2: (5, 0.12, 600, 250)}       # URLLC: 2000 / 5000 = 40 %; 600 B * 8 / 0.12 Mbps = 40 ms
IDLE_S, N_REPORTS = 30, 400            # traffic starts 30 s in, 100 s per trace


def _ue_file(path: Path, imsi: int, slice_id: int, t0: int) -> None:
    prb, rate, buf, granted = SPECS[slice_id]
    lines = [HEADER]
    for i in range(N_REPORTS):
        on = i * 0.25 >= IDLE_S
        lines.append(",".join(map(str, [
            t0 + 250 * i, 6, imsi, 70, "", 1, slice_id, prb, 1, 0, "",
            12 if on else 0, 30 if on else 0, buf if on else 0, rate if on else 0.0, 10, 0, 8.5, "",
            0, 0, 0, 0, 0, 0, 0, 0, 0, "",          # no uplink activity
            granted, granted if on else 0, "", 0, 0, 0, 0])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def raw_tree(tmp_path):
    root = tmp_path / "rome_static_medium"
    base = root / "sched0" / "tr9" / "exp1" / "bs1" / "slices_bs1"
    for k, sl in enumerate([0, 0, 1, 1, 2, 2]):
        imsi = 1010123456002 + k
        _ue_file(base / f"{imsi}_metrics.csv", imsi, sl, 1_617_000_000_000 + 1000 * k)
    return root


def test_records_are_native_telemetry(raw_tree):
    groups = prep.discover(raw_tree, None)
    assert list(groups) == [(0, 9, "exp1", 1)]
    rows = prep.build_cell_run((0, 9, "exp1", 1), groups[(0, 9, "exp1", 1)], window_s=1.0)
    records = [TelemetryRecord.model_validate(r) for r in rows]

    # idle start-up trimmed, three slices per window, one cell
    assert min(r.window_start_s for r in records) >= IDLE_S
    assert {r.slice_id for r in records} == {"eMBB", "MTC", "URLLC"}
    assert {r.cell_id for r in records} == {"bs1"}
    assert len(records) % 3 == 0

    mid = sorted({r.window_start_s for r in records})[10]       # every UE active
    by_slice = {r.slice_id: r for r in records if r.window_start_s == mid}
    embb, mtc, urllc = by_slice["eMBB"], by_slice["MTC"], by_slice["URLLC"]
    assert embb.scheduler == "round_robin" and embb.slice_prb == 30 and embb.cell_prb == 50
    assert embb.active_ues == 2
    assert np.isclose(embb.dl_throughput_per_ue_mbps, 3.0) and np.isclose(embb.dl_throughput_mbps, 6.0)
    assert np.isclose(embb.prb_utilisation, 1.0)
    assert np.isclose(mtc.prb_utilisation, 0.08) and np.isclose(urllc.prb_utilisation, 0.4)
    assert np.isclose(urllc.queueing_delay_ms, 40.0)
    assert mtc.queueing_delay_ms == 0.0
    # spare PRBs seen by eMBB: MTC 15 x 0.92 + URLLC 5 x 0.6
    assert np.isclose(embb.cell_spare_prb, 16.8)
    assert embb.ul_sinr_db is None            # no report carried uplink


def test_backlog_without_delivery_counts_as_full_window():
    d = pd.DataFrame({"w": 0, "slice": [0, 0, 1, 2], "imsi": ["a", "b", "c", "d"],
                      "rate": [0.0, 2.0, 0.1, 0.1], "buf": [5_000.0, 0.0, 0.0, 0.0],
                      "granted": 10, "cqi": 8.0, "mcs": 10.0, "sinr": np.nan})
    out = prep.slice_windows(d, pd.Series({0: 30, 1: 15, 2: 5}), window_s=1.0)
    embb = out[out.slice == 0].iloc[0]
    assert np.isclose(embb.queueing_delay_ms, (1000.0 + 0.0) / 2)   # stalled UE + idle UE


def test_csv_round_trips_through_the_runner(raw_tree, tmp_path, monkeypatch):
    out = tmp_path / "coloran.csv"
    monkeypatch.setattr(sys, "argv", ["prepare_coloran.py", "--root", str(raw_tree), "--out", str(out)])
    prep.main()
    df = pd.read_csv(out)
    assert list(df.columns) == list(TelemetryRecord.model_fields)   # no processing in the runner
    assert out.with_suffix(".summary.json").is_file()

    analyses = {}
    for _, row in df.iterrows():
        rec = prep.row_to_record(row)
        analyses[rec.slice_id] = MonitoringAgent.analyze_sla(rec).violations
    assert analyses == {"eMBB": ["throughput", "queueing_delay"],   # 3.0 < 3.6 Mbps, 133 ms > 100 ms
                        "MTC": [],
                        "URLLC": ["queueing_delay"]}               # 40 ms > 10 ms
