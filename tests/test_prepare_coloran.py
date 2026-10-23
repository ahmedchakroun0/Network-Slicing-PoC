"""Tests of scripts/prepare_coloran.py on a tiny synthetic tree that has the
exact ColO-RAN file layout and column names (no download needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_coloran as prep  # noqa: E402

from scripts.run_dataset import _row_to_telemetry_record  # noqa: E402
from agents.monitoring_agent import MonitoringAgent  # noqa: E402

COLS = ["Timestamp", "num_ues", "IMSI", "RNTI", "", "slicing_enabled", "slice_id", "slice_prb",
        "power_multiplier", "scheduling_policy", "", "dl_mcs", "dl_n_samples", "dl_buffer [bytes]",
        "tx_brate downlink [Mbps]", "tx_pkts downlink", "tx_errors downlink (%)", "dl_cqi"]


def _ue_file(path: Path, slice_id: int, t0: int, rate: float, buf: float, n: int = 400, idle_s: float = 30):
    rows = []
    for i in range(n):
        t = t0 + 250 * i
        active = i * 0.25 >= idle_s
        rows.append([t, 6, 1010123456000 + slice_id, 70, "", 1, slice_id, 6, 1, 0, "", 10, 30,
                     buf if active else 0, rate if active else 0.0, 10, 0, 12])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)


@pytest.fixture
def raw_tree(tmp_path):
    root = tmp_path / "rome_static_medium"
    base = root / "sched0" / "tr6" / "exp1" / "bs1" / "slices_bs1"
    # eMBB below budget (3 Mbps/UE), MTC fine, URLLC with a 40 ms queue -> latency violation
    specs = [(0, 3.0, 0), (0, 3.0, 0), (1, 0.06, 0), (1, 0.06, 0), (2, 0.12, 600), (2, 0.12, 600)]
    for k, (sl, rate, buf) in enumerate(specs):
        _ue_file(base / f"10101234560{k:02d}_metrics.csv", sl, 1_617_000_000_000 + 1000 * k, rate, buf)
    return root


def test_prepared_records_are_valid_telemetry(raw_tree):
    groups = prep.discover(raw_tree, None)
    assert list(groups) == [(0, 6, "exp1", 1)]
    rows = prep.build_cell_run((0, 6, "exp1", 1), groups[(0, 6, "exp1", 1)], window_s=1.0, warmup_s=20.0)
    df = prep.label(pd.DataFrame(rows))

    # start-up (idle) windows are removed: every record carries traffic
    assert (df["Slice Available Transfer Rate (Gbps)"] > 0).all()
    assert df["Window Start (s)"].min() >= 30.0
    # 3 slices per window, same cell
    assert set(df["Slice Type"]) == {"eMBB", "MTC", "URLLC"}
    assert set(df["Cell ID"]) == {"bs1"}
    assert len(df) % 3 == 0

    # steady state (all synthetic UEs active; they start 1 s apart)
    by_slice = df[df["Window Start (s)"] >= 40].groupby("Slice Type").first()
    assert by_slice.loc["eMBB", "Throughput Violation"]            # 6 Mbps < 7.2 Mbps budget
    assert np.isclose(by_slice.loc["eMBB", "Slice Available Transfer Rate (Gbps)"], 6e-3)
    assert not by_slice.loc["MTC", "SLA Evaluation Status"] == "violation"
    assert by_slice.loc["URLLC", "Latency Violation"]              # 600 B * 8 / 0.12 Mbps = 40 ms > 10 ms
    assert np.isclose(by_slice.loc["URLLC", "Slice Latency (ns)"], 40e6)

    # every row converts through the PoC loader and gets the same labels from analyze_sla
    df.insert(0, "record_id", [f"r{i}" for i in range(len(df))])
    for _, row in df.iterrows():
        rec = _row_to_telemetry_record(row, row["record_id"])
        assert rec.cell_id == "bs1"
        analysis = MonitoringAgent.analyze_sla(rec)
        assert ("latency" in analysis.violations) == bool(row["Latency Violation"])
        assert ("throughput" in analysis.violations) == bool(row["Throughput Violation"])


def test_prepared_csv_has_poc_columns(raw_tree, tmp_path, monkeypatch):
    out = tmp_path / "coloran.csv"
    monkeypatch.setattr(sys, "argv", ["prepare_coloran.py", "--root", str(raw_tree), "--out", str(out)])
    prep.main()
    df = pd.read_csv(out)
    original = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "prepared_network_slicing_dataset.csv",
                           nrows=1)
    assert set(original.columns) <= set(df.columns)          # drop-in replacement
    assert "Cell ID" in df.columns
    assert out.with_suffix(".summary.json").is_file()
