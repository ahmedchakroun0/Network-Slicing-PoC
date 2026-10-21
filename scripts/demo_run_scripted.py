"""Dry run of scripts/run_dataset.py WITHOUT a model, Redis or MongoDB.

Uses the scripted tool-calling LLM of tests/test_end_to_end.py, fakeredis and
mongomock, so the whole pipeline (Monitoring -> RIC agents -> middleware ->
governance -> audit) can be checked on any prepared CSV before running a
real model.  Usage:
    poetry run python scripts/demo_run_scripted.py 60 [path/to/prepared.csv]
The scripted LLM always selects both RIC agents: routing quality is NOT
evaluated here, only that the data flows through every component.
"""
import asyncio, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import os; os.chdir(ROOT)
import fakeredis.aioredis, mongomock
import agents.base_ric as base_ric, scripts.run_dataset as rd, src.core.llm as core_llm
import src.repositories.mongo_repository as mrepo
from src.memory.working_memory import RedisWorkingMemory
from test_end_to_end import ScriptedChatModel
m = ScriptedChatModel()
rd.create_llm = base_ric.create_llm = core_llm.create_llm = (lambda: m)
mrepo.MongoClient = mongomock.MongoClient
redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
RedisWorkingMemory.from_url = classmethod(lambda cls, *a, **k: cls(redis))
store = {}
orig = mrepo.MongoRepository.__post_init__
def cap(self): orig(self); store["c"] = self._collection
mrepo.MongoRepository.__post_init__ = cap
cfg = rd.RunConfig(dataset_path=Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/coloran_prepared_dataset.csv", run_id="coloran-demo",
                   start_row=1, max_rows=int(sys.argv[1]), knowledge_file=rd.DEFAULT_KNOWLEDGE_FILE)
asyncio.run(rd.run_dataset(cfg))
doc = store["c"].find_one({"_id": "coloran-demo"})
print("SUMMARY", json.dumps(doc["summary"]))
for rid, r in list(doc["records"].items()):
    t = r["telemetry_record"]; d = r["monitoring_decision"] or {}
    ex = r.get("agent_executions") or {}
    out = {a: ((e.get("middleware_audit") or [{}])[0].get("governance") or {}).get("failed_stage") or "approved"
           for a, e in ex.items()}
    print(rid, t["cell_id"], t["slice_type"], r["computed_analysis"]["violations"],
          "agents=", [i["agent"] for i in d.get("selected_agents", [])], out,
          "persist=", (ex.get("non_rt_ric", {}).get("invocation", {}).get("context", {}).get("persistence")))
