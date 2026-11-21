import pytest

from ran_emulator import overlay
from ran_emulator.engine import Emulator
from ran_emulator.scenarios import EventSpec


def test_neutral_radio_effect_changes_nothing():
    e = overlay.radio_effect(9.0, 15.0, 0.0, 0, overlay.BASE_BLER, 0.0)
    assert (e.cqi, e.mcs) == (9.0, 15.0)
    assert e.rate_factor == pytest.approx(1.0)
    assert overlay.harq_extra_delay_ms(e.bler) == pytest.approx(0.0)


def test_more_power_raises_cqi_mcs_and_capacity():
    e = overlay.radio_effect(9.0, 15.0, 6.0, 0, overlay.BASE_BLER, 0.0)
    assert e.cqi == 12.0 and e.mcs == 21.0 and e.rate_factor > 1.3


def test_link_adaptation_lag_raises_bler_and_a_lower_mcs_fixes_it():
    lagging = overlay.radio_effect(9.0, 15.0, -8.0, 0, overlay.BASE_BLER, 8.0)
    fixed = overlay.radio_effect(9.0, 15.0, -8.0, -4, overlay.BASE_BLER, 8.0)
    assert overlay.harq_extra_delay_ms(lagging.bler) > 10
    assert fixed.bler == pytest.approx(overlay.BASE_BLER)


def test_handover_margin_model():
    assert overlay.handover_sinr_db(3.0) == 0.0
    assert overlay.handover_sinr_db(2.0) > 0
    assert overlay.handover_sinr_db(0.0) < 0


def _embb(emu, steps):
    return [next(r for r in emu.step().records if r.slice_id == "eMBB") for _ in range(steps)]


def test_demand_surge_builds_delay_then_recovers(bank, make_scenario):
    ev = EventSpec(type="demand_surge", cell="bs2", slice="eMBB", factor=1.5,
                   start_step=3, duration=5)
    emu = Emulator(bank, make_scenario(tr=12, events=[ev]))
    recs = _embb(emu, 20)
    assert recs[5].queueing_delay_ms > recs[0].queueing_delay_ms + 50
    assert recs[-1].queueing_delay_ms == pytest.approx(recs[0].queueing_delay_ms, abs=1)


def test_surge_demand_is_anchored_so_more_capacity_absorbs_it(bank, make_scenario):
    ev = EventSpec(type="demand_surge", cell="bs2", slice="eMBB", factor=1.2,
                   start_step=0, duration=50)
    emu = Emulator(bank, make_scenario(tr=10, events=[ev]))       # eMBB 24 PRBs
    congested = _embb(emu, 3)[-1].queueing_delay_ms
    emu.set_prb("bs2", "eMBB", 84)                                # 42 PRBs
    assert _embb(emu, 5)[-1].queueing_delay_ms < congested / 2


def test_interference_is_compensated_by_power(bank, make_scenario):
    ev = EventSpec(type="interference", cell="bs2", db=6, start_step=0, duration=50)
    emu = Emulator(bank, make_scenario(tr=12, events=[ev]))
    hit = _embb(emu, 3)[-1]
    emu.set_power("bs2", 6)
    fixed = _embb(emu, 6)[-1]
    assert hit.dl_cqi == 6.0 and fixed.dl_cqi == 9.0
    assert fixed.queueing_delay_ms < hit.queueing_delay_ms


def test_outage_zeroes_service_then_drains(bank, make_scenario):
    ev = EventSpec(type="cell_outage", cell="bs2", start_step=2, duration=3)
    emu = Emulator(bank, make_scenario(tr=12, events=[ev]))
    recs = _embb(emu, 15)
    assert [r.dl_throughput_mbps for r in recs[2:5]] == [0.0, 0.0, 0.0]
    assert recs[3].queueing_delay_ms == 1000.0
    assert emu.cells["bs2"].slices["eMBB"].backlog_bytes == 0.0


def test_priority_borrows_idle_prbs(bank, make_scenario):
    ev = EventSpec(type="demand_surge", cell="bs2", slice="eMBB", factor=1.3,
                   start_step=0, duration=50)
    emu = Emulator(bank, make_scenario(tr=12, events=[ev]))
    before = _embb(emu, 3)[-1]
    emu.set_qos("bs2", "eMBB", 1, 50, 4)
    after = _embb(emu, 5)[-1]
    assert after.dl_throughput_mbps > before.dl_throughput_mbps


def test_cell_spare_prb_is_recomputed_when_modified(bank, make_scenario):
    ev = EventSpec(type="cell_outage", cell="bs2", start_step=0, duration=1)
    recs = Emulator(bank, make_scenario(tr=12, events=[ev])).step().records
    # Every slice idle: spare PRBs of a slice = all PRBs of the other two.
    assert {r.slice_id: r.cell_spare_prb for r in recs} == {"eMBB": 14, "MTC": 47, "URLLC": 39}
