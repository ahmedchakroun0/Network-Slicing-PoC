"""Dry run of scripts/run_loop.py WITHOUT a model, Redis, MongoDB, LiteLLM or
a running emulator.

Uses the scripted tool-calling LLM of tests/test_end_to_end.py, fakeredis,
mongomock, the emulator app in process (real ColO-RAN traces: the full
prepared dataset, else the tracked sample) and in-process MCP clients, so the
whole closed loop (emulator -> Monitoring -> RIC agents -> middleware ->
governance -> MCP tools -> emulator -> verification) can be checked before
running a real model. Usage:
    poetry run python scripts/demo_run_scripted.py 10 [scenario]
The scripted LLM always selects both RIC agents and always asks for a 72 %
PRB share: routing and action quality are NOT evaluated here, only that the
data flows through every component.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ["LANGSMITH_TRACING"] = "false"  # scripted data: no traces

import fakeredis.aioredis
import httpx
import mongomock
from mcp import Client
from test_end_to_end import ScriptedChatModel

import agents.base_ric as base_ric
import scripts.run_loop as rl
import src.core.llm as core_llm
import src.repositories.mongo_repository as mrepo
from ran_emulator.api import create_app
from ran_emulator.client import EmulatorClient
from ran_emulator.scenarios import load_scenario
from ran_emulator.traces import TraceBank
from ric_tools.server import build_server
from src.memory.working_memory import RedisWorkingMemory

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
scenario = sys.argv[2] if len(sys.argv) > 2 else "embb_prb_starvation"

m = ScriptedChatModel()
rl.create_llm = base_ric.create_llm = core_llm.create_llm = (lambda: m)
mrepo.MongoClient = mongomock.MongoClient
redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
RedisWorkingMemory.from_url = classmethod(lambda cls, *a, **k: cls(redis))
store = {}
orig = mrepo.MongoRepository.__init__


def cap(self, *a, **k):
    orig(self, *a, **k)
    store["repo"] = self


mrepo.MongoRepository.__init__ = cap


async def main() -> None:
    app = create_app(TraceBank.load(), load_scenario(scenario))
    emulator = EmulatorClient(base_url="http://emu", transport=httpx.ASGITransport(app=app))
    async with Client(build_server("near_rt_ric", emulator)) as near, \
            Client(build_server("non_rt_ric", emulator)) as non:
        await rl.run_loop(rl.RunConfig(run_id="demo", scenario=scenario, steps=steps),
                          emulator=emulator,
                          mcp_clients={"near_rt_ric": near, "non_rt_ric": non})
        print("EMULATOR ACTIONS", json.dumps(await emulator._call("GET", "/sim/actions")))


asyncio.run(main())
repo = store["repo"]
print("RUN", json.dumps(repo.runs.find_one({"_id": "demo"})["summary"]))
for r in repo.records.find({"run_id": "demo"}).sort("seq", 1):
    t, d = r["telemetry"], r.get("monitoring") or {}
    out = {a: [(x["governance"].get("failed_stage") or "approved",
                (x.get("outcome") or {}).get("outcome")) for x in e.get("actions", [])]
           for a, e in (r.get("agents") or {}).items()}
    print(r["seq"], t["cell_id"], t["slice_id"], r["sla"].get("violations"), "persist=",
          r["persistence"], "agents=", [i["agent"] for i in d.get("selected_agents", [])], out)
print("REDIS KEYS", sorted(asyncio.run(redis.keys("wm:*"))))
