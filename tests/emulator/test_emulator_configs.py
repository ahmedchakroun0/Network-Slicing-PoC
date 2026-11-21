import pytest

from ran_emulator.configs import CELL_PRB, SPLITS, Split, snap
from ran_emulator.scenarios import SCENARIO_DIR, load_scenario


def test_the_28_recorded_splits_in_coloran_order():
    assert len(SPLITS) == 28
    assert all(s.embb + s.mtc + s.urllc == CELL_PRB and s.mtc >= 3 for s in SPLITS)
    # data/exploration/07_config_slice_prb.csv
    assert SPLITS[0] == Split(6, 39, 5)
    assert SPLITS[12] == Split(36, 3, 11)
    assert SPLITS[27] == Split(6, 3, 41)


def test_snap_moves_the_target_slice_and_keeps_the_others_close():
    assert snap(SPLITS[9], "eMBB", 36, {}) == Split(36, 3, 11)   # URLLC kept at 11
    assert snap(SPLITS[9], "URLLC", 17, {}) == Split(18, 15, 17)  # eMBB kept at 18


def test_snap_respects_policies_and_fails_when_none_fit():
    got = snap(SPLITS[9], "eMBB", 42, {"URLLC": (20, 100)})        # URLLC >= 10 PRBs
    assert got.urllc >= 10
    with pytest.raises(ValueError):
        snap(SPLITS[9], "eMBB", 6, {"eMBB": (80, 100), "URLLC": (80, 100)})


def test_shipped_scenarios_are_valid():
    paths = sorted(SCENARIO_DIR.glob("*.json"))
    assert len(paths) >= 6
    for p in paths:
        assert load_scenario(p.stem).name == p.stem
