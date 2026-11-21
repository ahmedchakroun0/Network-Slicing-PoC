import httpx
import pytest
from fastapi.testclient import TestClient

from ran_emulator.api import create_app
from ran_emulator.client import EmulatorClient, EmulatorError
from src.schema.monitoring import TelemetryRecord


@pytest.fixture
def client(bank, make_scenario):
    with TestClient(create_app(bank, make_scenario(tr=9))) as c:
        yield c


def test_run_returns_valid_records_and_meta(client):
    steps = client.post("/sim/run", json={"steps": 2}).json()["steps"]
    assert [s["step"] for s in steps] == [0, 1]
    for rec in steps[0]["records"]:
        TelemetryRecord.model_validate(rec)
    assert steps[0]["meta"]["bs2"]["source_run"] == "sched0/tr9/exp1"


@pytest.mark.parametrize("path, body", [
    ("/cells/bs2/slices/eMBB/prb", {"prb_percentage": 72}),
    ("/cells/bs2/scheduler", {"scheduler": "proportional_fair"}),
    ("/cells/bs2/handover", {"handover_margin_db": 2}),
    ("/cells/bs2/power", {"power_offset_db": 3}),
    ("/cells/bs2/slices/URLLC/qos", {"priority_level": 2, "packet_delay_budget_ms": 10,
                                     "guaranteed_bitrate_mbps": 0.2}),
    ("/cells/bs2/slices/URLLC/link-adaptation", {"mcs_offset": -2, "target_bler": 0.01}),
    ("/cells/bs2/slices/eMBB/policy", {"min_prb_ratio_pct": 20, "max_prb_ratio_pct": 90}),
])
def test_every_control_reports_previous_current_and_effective_step(client, path, body):
    client.post("/sim/run", json={"steps": 3})
    r = client.put(path, json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert {"previous", "current"} <= out.keys() and out["effective_step"] == 3


def test_out_of_range_is_422_and_logged(client):
    r = client.put("/cells/bs2/power", json={"power_offset_db": 20})
    assert r.status_code == 422
    assert client.get("/sim/violations").json()[0]["control"] == "power"


def test_unknown_cell_or_slice(client):
    assert client.put("/cells/bs9/power", json={"power_offset_db": 1}).status_code == 404
    assert client.put("/cells/bs2/slices/IoT/prb", json={"prb_percentage": 30}).status_code == 422


def test_reset_events_state_and_kpis(client):
    client.post("/sim/run", json={"steps": 2})
    assert client.post("/sim/reset", json={"seed": 3}).json()["next_step"] == 0
    ev = client.post("/sim/events", json={"type": "cell_outage", "cell": "bs2", "duration": 2})
    assert ev.status_code == 200
    step = client.post("/sim/run", json={"steps": 1}).json()["steps"][0]
    assert all(r["dl_throughput_mbps"] == 0 for r in step["records"])
    assert len(client.get("/cells/bs2/kpis", params={"from_step": 0}).json()) == 3
    assert client.get("/state").json()["mode"] == "lockstep"
    assert client.post("/sim/reset", json={"scenario": "nope"}).status_code == 404


def test_realtime_mode_refuses_manual_steps(bank, make_scenario):
    app = create_app(bank, make_scenario(), mode="realtime", step_period_s=3600)
    with TestClient(app) as c:
        assert c.post("/sim/run", json={"steps": 1}).status_code == 409


async def test_async_client(bank, make_scenario):
    app = create_app(bank, make_scenario(tr=9))
    async with EmulatorClient("http://emu", transport=httpx.ASGITransport(app=app)) as emu:
        out = await emu.adjust_prb_allocation("bs2", "eMBB", 72)
        assert out["current"]["eMBB"] == 36
        records = await emu.step_records()
        assert next(r for r in records if r.slice_id == "eMBB").slice_prb == 36
        with pytest.raises(EmulatorError) as exc:
            await emu.adjust_power_control("bs2", 99)
        assert exc.value.status_code == 422
        assert (await emu.state())["next_step"] == 1
