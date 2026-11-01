"""MongoDB results store (src/repositories/mongo_repository.py) on mongomock."""

from datetime import datetime, timezone

from src.schema.mongo_run_document import ActionEntry, AgentMessage, KnowledgeEntry


def _action(i, approved):
    return ActionEntry(
        i=i, tool_call_id=f"call_{i}", at=datetime.now(timezone.utc), tool="adjust_prb_allocation",
        target="bs1", params={"slice_id": "eMBB", "prb_percentage": 60},
        reasoning={"rationale": "throughput 1.8 < 3.6 Mbps", "expected_effect": "throughput up",
                   "confidence": 0.7},
        agent_message=AgentMessage(text="Raising the PRB share."),
        governance={"approved": approved, "failed_stage": None if approved else "llm_review",
                    "reason": "ok" if approved else "rejected",
                    "llm_review": {"goal_aligned": True, "reasoning_sound": approved,
                                   "behavior_acceptable": True, "reason": "why"}},
        execution={"status": "executed" if approved else "rejected", "result": {}, "message": None},
    )


def _insert(mongo, record_id="r1", seq=1):
    mongo.insert_record(run_id="run1", record_id=record_id, seq=seq,
                        telemetry={"cell_id": "bs1", "slice_id": "eMBB"},
                        sla={"violations": ["throughput"]}, persistence={"throughput": 1})


def test_run_document_is_small_and_counted(mongo):
    mongo.start_run(run_id="run1", config={"model": "m"}, summary={"rows_processed": 0})
    mongo.count(run_id="run1", counters={"rows_processed": 1, "agent_invocations.near_rt_ric": 1})
    mongo.count(run_id="run1", counters={"rows_processed": 1})
    mongo.finish_run(run_id="run1")
    run = mongo.runs.find_one({"_id": "run1"})
    assert run["summary"]["rows_processed"] == 2
    assert run["summary"]["agent_invocations"]["near_rt_ric"] == 1
    assert run["status"] == "finished" and run["finished_at"] is not None
    assert "records" not in run                        # records live in their own collection


def test_each_step_adds_to_the_record_document(mongo):
    _insert(mongo)
    mongo.set_decision(run_id="run1", record_id="r1",
                       decision={"situation": "critical", "reason": "throughput below budget"})
    mongo.start_agent(run_id="run1", record_id="r1", agent="near_rt_ric")
    mongo.push_knowledge(run_id="run1", record_id="r1", agent="near_rt_ric",
                         entry=KnowledgeEntry(tool_call_id="k1", at=datetime.now(timezone.utc),
                                              tool="search_ran_knowledge", args={"query": "q"},
                                              result="snippet"))
    mongo.push_action(run_id="run1", record_id="r1", agent="near_rt_ric", entry=_action(0, False))
    mongo.push_action(run_id="run1", record_id="r1", agent="near_rt_ric", entry=_action(1, True))
    mongo.finish_agent(run_id="run1", record_id="r1", agent="near_rt_ric", status="completed",
                       selected_action=1, final_message=AgentMessage(text="Done."))
    mongo.finish_record(run_id="run1", record_id="r1")

    doc = mongo.records.find_one({"_id": "run1:r1"})
    assert doc["status"] == "done" and doc["seq"] == 1
    assert doc["telemetry"]["cell_id"] == "bs1" and doc["sla"]["violations"] == ["throughput"]
    assert doc["monitoring"]["reason"] == "throughput below budget"
    agent = doc["agents"]["near_rt_ric"]
    assert agent["status"] == "completed" and agent["selected_action"] == 1
    assert [a["execution"]["status"] for a in agent["actions"]] == ["rejected", "executed"]
    assert agent["actions"][0]["governance"]["llm_review"]["reasoning_sound"] is False
    assert agent["actions"][0]["reasoning"]["rationale"].startswith("throughput")
    assert agent["knowledge"][0]["result"] == "snippet"
    assert agent["final_message"]["text"] == "Done."


def test_insert_record_never_overwrites(mongo):
    _insert(mongo)
    mongo.set_decision(run_id="run1", record_id="r1", decision={"reason": "kept"})
    mongo.records.update_one({"_id": "run1:r1"}, {"$set": {"marker": 1}})
    _insert(mongo)                                        # same record again
    doc = mongo.records.find_one({"_id": "run1:r1"})
    assert doc["monitoring"]["reason"] == "kept" and doc["marker"] == 1


def test_fail_record_creates_or_marks_the_record(mongo):
    mongo.fail_record(run_id="run1", record_id="bad", seq=7, error={"type": "ValueError", "message": "x"})
    doc = mongo.records.find_one({"_id": "run1:bad"})
    assert doc["status"] == "failed" and doc["seq"] == 7 and doc["error"]["type"] == "ValueError"

    _insert(mongo, "r2", 8)
    mongo.fail_record(run_id="run1", record_id="r2", seq=8, error={"type": "KeyError", "message": "y"})
    doc = mongo.records.find_one({"_id": "run1:r2"})
    assert doc["status"] == "failed" and doc["telemetry"]["cell_id"] == "bs1"


def test_records_are_indexed_for_run_queries(mongo):
    keys = [tuple(k for k, _ in idx["key"]) for idx in mongo.records.index_information().values()]
    assert ("run_id", "seq") in keys
    assert ("run_id", "telemetry.cell_id", "telemetry.slice_id") in keys
