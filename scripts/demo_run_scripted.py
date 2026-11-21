"""Dry run of scripts/run_dataset.py WITHOUT a model, Redis or MongoDB.

Uses the scripted tool-calling LLM of tests/test_end_to_end.py, fakeredis and
mongomock, so the whole pipeline (Monitoring -> RIC agents -> middleware ->
governance -> audit) can be checked on any prepared CSV before running a
real model.  Usage:
    poetry run python scripts/demo_run_scripted.py 60 [path/to/prepared.csv]
The scripted LLM always selects both RIC agents: routing quality is NOT
evaluated here, only that the data flows through every component.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.chdir(ROOT)
os.environ["LANGSMITH_TRACING"] = "false"  # scripted data: no traces

import fakeredis.aioredis
import mongomock
from test_end_to_end import ScriptedChatModel

import agents.base_ric as base_ric
import scripts.run_dataset as rd
import src.core.llm as core_llm
import src.repositories.mongo_repository as mrepo
from src.memory.working_memory import RedisWorkingMemory

m = ScriptedChatModel()
rd.create_llm = base_ric.create_llm = core_llm.create_llm = (lambda: m)
mrepo.MongoClient = mongomock.MongoClient
redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
RedisWorkingMemory.from_url = classmethod(lambda cls, *a, **k: cls(redis))
store = {}
orig = mrepo.MongoRepository.__init__


def cap(self, *a, **k):
    orig(self, *a, **k)
    store["repo"] = self


mrepo.MongoRepository.__init__ = cap
cfg = rd.RunConfig(dataset_path=Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/coloran_prepared_dataset.csv", run_id="coloran-demo",
                   start_row=1, max_rows=int(sys.argv[1]), knowledge_file=rd.DEFAULT_KNOWLEDGE_FILE)
asyncio.run(rd.run_dataset(cfg))
repo = store["repo"]
print("RUN", json.dumps(repo.runs.find_one({"_id": "coloran-demo"})["summary"]))
for r in repo.records.find({"run_id": "coloran-demo"}).sort("seq", 1):
    t, d = r["telemetry"], r.get("monitoring") or {}
    out = {a: [x["governance"].get("failed_stage") or "approved" for x in e.get("actions", [])]
           for a, e in (r.get("agents") or {}).items()}
    print(r["seq"], t["cell_id"], t["slice_id"], r["sla"].get("violations"), "persist=", r["persistence"],
          "agents=", [i["agent"] for i in d.get("selected_agents", [])], out,
          "bytes=", len(json.dumps(r, default=str)))
print("REDIS KEYS", sorted(asyncio.run(redis.keys("wm:*"))))
