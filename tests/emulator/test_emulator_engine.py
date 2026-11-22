from pathlib import Path

import pandas as pd
import pytest

from ran_emulator.configs import SPLITS
from ran_emulator.engine import ControlError, Emulator
from ran_emulator.scenarios import CellSpec, EventSpec, Scenario
from ran_emulator.traces import KPI_FIELDS, TraceBank

SAMPLE = Path(__file__).resolve().parents[2] / "data/coloran_prepared_sample.csv"


def _by_slice(result):
    return {r.slice_id: r for r in result.records}


def _assert_replay(emu: Emulator, df: pd.DataFrame, steps: int) -> None:
    for _ in range(steps):
        res = emu.step()
        for rec in res.records:
            meta = res.meta[rec.cell_id]
            src = df[(df.run == meta["source_run"]) & (df.cell_id == rec.cell_id)
                     & (df.slice_id == rec.slice_id)].sort_values("window_start_s")
            row = src.iloc[meta["trace_offset"]]
            for f in (*KPI_FIELDS, "slice_prb", "scheduler"):
                expected = None if pd.isna(row[f]) else row[f]
                assert getattr(rec, f) == expected, f


def test_transparent_replay_without_actions_or_events(bank, frame, make_scenario):
    emu = Emulator(bank, make_scenario(tr=9))
    _assert_replay(emu, frame, steps=25)      # also crosses the end of the trace


@pytest.mark.skipif(not SAMPLE.exists(), reason="prepared ColO-RAN sample not present")
def test_transparent_replay_on_real_coloran_sample():
    df = pd.read_csv(SAMPLE)
    bank = TraceBank.from_frame(df)
    trace = next(iter(bank))
    sc = Scenario(name="real", cells={trace.cell_id: CellSpec(
        scheduler=trace.scheduler, tr=trace.split.tr, start_offset=0)})
    _assert_replay(Emulator(bank, sc), df, steps=10)


def test_records_are_valid_telemetry_with_emulator_ids(bank, make_scenario):
    res = Emulator(bank, make_scenario()).step()
    assert [r.slice_id for r in res.records] == ["eMBB", "MTC", "URLLC"]
    assert res.records[0].record_id == "emu-t-bs2-s00000-eMBB"
    assert res.records[0].run == "emu/t/seed0"


def test_prb_change_switches_trace_at_next_step(bank, make_scenario):
    emu = Emulator(bank, make_scenario(tr=9))
    before = _by_slice(emu.step())["eMBB"]
    out = emu.set_prb("bs2", "eMBB", 72)
    assert out["previous"] == {"eMBB": 18, "MTC": 21, "URLLC": 11}
    assert out["current"] == {"eMBB": 36, "MTC": 3, "URLLC": 11}
    assert out["effective_step"] == 1
    after = _by_slice(emu.step())["eMBB"]
    assert after.slice_prb == 36
    assert after.dl_throughput_mbps > before.dl_throughput_mbps * 1.9


def test_prb_request_is_snapped_and_reported(bank, make_scenario):
    out = Emulator(bank, make_scenario(tr=9)).set_prb("bs2", "eMBB", 50)
    assert out["applied_pct"] == 48.0 and "snapped" in out["note"]


def test_scheduler_change_applies_to_the_cell(bank, make_scenario):
    emu = Emulator(bank, make_scenario(tr=9))
    base = _by_slice(emu.step())["eMBB"].dl_throughput_mbps
    out = emu.set_scheduler("bs2", "proportional_fair")
    assert "one scheduler per cell" in out["note"]
    rec = _by_slice(emu.step())["eMBB"]
    assert rec.scheduler == "proportional_fair" and rec.dl_throughput_mbps > base


def test_missing_trace_falls_back_to_nearest_and_says_so(bank, make_scenario):
    emu = Emulator(bank, make_scenario(tr=9))
    out = emu.set_scheduler("bs2", "waterfilling")      # not in the synthetic bank
    assert "nearest" in out["note"]
    assert emu.cells["bs2"].scheduler == "round_robin"


@pytest.mark.parametrize("call", [
    lambda e: e.set_prb("bs2", "eMBB", 95),
    lambda e: e.set_power("bs2", 9),
    lambda e: e.set_handover("bs2", -1),
    lambda e: e.set_scheduler("bs2", "fifo"),
    lambda e: e.set_link_adaptation("bs2", "URLLC", -8, 0.1),
    lambda e: e.set_qos("bs2", "eMBB", 0),
    lambda e: e.set_policy("bs2", "eMBB", 60, 40),
])
def test_out_of_envelope_controls_are_refused_and_logged(bank, make_scenario, call):
    emu = Emulator(bank, make_scenario())
    with pytest.raises(ControlError):
        call(emu)
    assert len(emu.violations) == 1 and emu.actions == []


def test_policy_moves_a_split_that_breaks_it(bank, make_scenario):
    emu = Emulator(bank, make_scenario(tr=9))           # eMBB 18 PRBs = 36 %
    out = emu.set_policy("bs2", "eMBB", 60, 90)
    assert out["prb"]["eMBB"] >= 30 and "violated the policy" in out["note"]
    # Later PRB actions are snapped inside the policy...
    assert emu.set_prb("bs2", "eMBB", 12)["current"]["eMBB"] >= 30
    # ...and a policy no recorded split can satisfy is refused.
    with pytest.raises(ControlError):
        emu.set_policy("bs2", "URLLC", 80, 100)


def test_same_seed_same_trajectory(bank, make_scenario):
    runs = []
    for _ in range(2):
        emu = Emulator(bank, make_scenario(start_offset=None, seed=7))
        runs.append([r.model_dump() for _ in range(5) for r in emu.step().records])
    assert runs[0] == runs[1]


def test_unknown_cell(bank, make_scenario):
    with pytest.raises(KeyError):
        Emulator(bank, make_scenario()).set_power("bs9", 1)


def test_history_and_runtime_events(bank, make_scenario):
    emu = Emulator(bank, make_scenario(tr=12))
    emu.step()
    emu.add_event(EventSpec(type="cell_outage", cell="bs2", start_step=1, duration=1))
    assert _by_slice(emu.step())["eMBB"].dl_throughput_mbps == 0
    assert [r["step"] for r in emu.kpis("bs2", from_step=1)] == [1, 1, 1]
    assert emu.state()["next_step"] == 2


def test_initial_split_outside_scenario_policy_is_snapped(bank):
    sc = Scenario(name="p", cells={"bs2": CellSpec(tr=9, start_offset=0,
                                                   policies={"eMBB": (60, 90)})})
    assert Emulator(bank, sc).cells["bs2"].split.embb >= 30
    assert SPLITS[9].embb == 18
