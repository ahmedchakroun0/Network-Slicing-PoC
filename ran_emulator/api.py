"""HTTP interface of the RAN emulator (FastAPI).

Control endpoints mirror the agents' action tools (tools/registry.py); the
MCP tool servers will call them through ran_emulator.client.EmulatorClient.

Clock modes:
  lockstep  - the clock only advances on POST /sim/run (deterministic experiments)
  realtime  - a background task advances one step every step_period_s
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ran_emulator.engine import ControlError, Emulator, StepResult
from ran_emulator.scenarios import EventSpec, Scenario, load_scenario
from ran_emulator.traces import TraceBank
from src.schema.sla import SliceType


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PRBBody(_Body):
    prb_percentage: float


class SchedulerBody(_Body):
    scheduler: str
    slice_id: SliceType | None = None


class HandoverBody(_Body):
    handover_margin_db: float


class PowerBody(_Body):
    power_offset_db: float


class QoSBody(_Body):
    priority_level: int
    packet_delay_budget_ms: float
    guaranteed_bitrate_mbps: float


class LinkAdaptationBody(_Body):
    mcs_offset: int
    target_bler: float


class PolicyBody(_Body):
    min_prb_ratio_pct: float
    max_prb_ratio_pct: float


class ResetBody(_Body):
    scenario: str | None = Field(default=None, description="Name or path; default: current.")
    seed: int | None = None


class RunBody(_Body):
    steps: int = Field(default=1, ge=1, le=10_000)


def _step_payload(r: StepResult) -> dict[str, Any]:
    return {"step": r.step, "records": [x.model_dump() for x in r.records], "meta": r.meta}


class EmulatorService:
    """One emulator instance plus its clock, lock and SSE subscribers."""

    def __init__(self, bank: TraceBank, scenario: Scenario, mode: Literal["lockstep", "realtime"],
                 step_period_s: float):
        self.bank = bank
        self.emu = Emulator(bank, scenario)
        self.mode = mode
        self.step_period_s = step_period_s
        self.lock = asyncio.Lock()
        self.subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    async def advance(self, steps: int) -> list[dict[str, Any]]:
        async with self.lock:
            out = [_step_payload(self.emu.step()) for _ in range(steps)]
        for payload in out:
            for q in self.subscribers:
                if q.full():                    # slow consumer: drop its oldest event
                    q.get_nowait()
                q.put_nowait(payload)
        return out

    async def control(self, fn: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
        async with self.lock:
            try:
                return fn(*args)
            except ControlError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc

    async def realtime_loop(self) -> None:
        while True:
            await asyncio.sleep(self.step_period_s)
            await self.advance(1)


def create_app(bank: TraceBank, scenario: Scenario, *,
               mode: Literal["lockstep", "realtime"] = "lockstep",
               step_period_s: float = 1.0) -> FastAPI:
    svc = EmulatorService(bank, scenario, mode, step_period_s)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(svc.realtime_loop()) if mode == "realtime" else None
        yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="RAN emulator (ColO-RAN traces)", lifespan=lifespan)
    app.state.service = svc

    # ---------------------------------------------------------- controls
    @app.put("/cells/{cell}/slices/{slice_id}/prb")
    async def prb(cell: str, slice_id: SliceType, body: PRBBody) -> dict[str, Any]:
        return await svc.control(svc.emu.set_prb, cell, slice_id, body.prb_percentage)

    @app.put("/cells/{cell}/scheduler")
    async def scheduler(cell: str, body: SchedulerBody) -> dict[str, Any]:
        return await svc.control(svc.emu.set_scheduler, cell, body.scheduler, body.slice_id)

    @app.put("/cells/{cell}/handover")
    async def handover(cell: str, body: HandoverBody) -> dict[str, Any]:
        return await svc.control(svc.emu.set_handover, cell, body.handover_margin_db)

    @app.put("/cells/{cell}/power")
    async def power(cell: str, body: PowerBody) -> dict[str, Any]:
        return await svc.control(svc.emu.set_power, cell, body.power_offset_db)

    @app.put("/cells/{cell}/slices/{slice_id}/qos")
    async def qos(cell: str, slice_id: SliceType, body: QoSBody) -> dict[str, Any]:
        return await svc.control(svc.emu.set_qos, cell, slice_id, body.priority_level,
                                 body.packet_delay_budget_ms, body.guaranteed_bitrate_mbps)

    @app.put("/cells/{cell}/slices/{slice_id}/link-adaptation")
    async def link_adaptation(cell: str, slice_id: SliceType,
                              body: LinkAdaptationBody) -> dict[str, Any]:
        return await svc.control(svc.emu.set_link_adaptation, cell, slice_id, body.mcs_offset,
                                 body.target_bler)

    @app.put("/cells/{cell}/slices/{slice_id}/policy")
    async def policy(cell: str, slice_id: SliceType, body: PolicyBody) -> dict[str, Any]:
        return await svc.control(svc.emu.set_policy, cell, slice_id, body.min_prb_ratio_pct,
                                 body.max_prb_ratio_pct)

    # ------------------------------------------------------ simulation
    @app.post("/sim/reset")
    async def reset(body: ResetBody) -> dict[str, Any]:
        try:
            sc = load_scenario(body.scenario) if body.scenario else svc.emu.scenario
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if body.seed is not None:
            sc = sc.model_copy(update={"seed": body.seed})
        async with svc.lock:
            try:
                svc.emu = Emulator(svc.bank, sc)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
            return svc.emu.state()

    @app.post("/sim/run")
    async def run(body: RunBody) -> dict[str, Any]:
        if svc.mode == "realtime":
            raise HTTPException(status_code=409, detail="the clock runs by itself in realtime mode")
        return {"steps": await svc.advance(body.steps)}

    @app.post("/sim/events")
    async def add_event(event: EventSpec) -> dict[str, Any]:
        async with svc.lock:
            if event.start_step < svc.emu.step_index:
                event = event.model_copy(update={"start_step": svc.emu.step_index})
            try:
                return svc.emu.add_event(event).model_dump()
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc

    @app.get("/sim/violations")
    async def violations() -> list[dict[str, Any]]:
        return svc.emu.violations

    @app.get("/sim/actions")
    async def actions() -> list[dict[str, Any]]:
        return svc.emu.actions

    # ----------------------------------------------------- observation
    @app.get("/state")
    async def state() -> dict[str, Any]:
        return {**svc.emu.state(), "mode": svc.mode}

    @app.get("/cells/{cell}/kpis")
    async def kpis(cell: str, from_step: int = 0, to_step: int | None = None) -> list[dict[str, Any]]:
        try:
            return svc.emu.kpis(cell, from_step, to_step)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc

    @app.get("/telemetry/stream")
    async def stream(request: Request) -> StreamingResponse:
        """Server-sent events, one `step` event per emulator step."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        svc.subscribers.add(q)

        async def events() -> AsyncIterator[str]:
            try:
                while not await request.is_disconnected():
                    try:
                        payload = await asyncio.wait_for(q.get(), timeout=15)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"event: step\ndata: {json.dumps(payload)}\n\n"
            finally:
                svc.subscribers.discard(q)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
