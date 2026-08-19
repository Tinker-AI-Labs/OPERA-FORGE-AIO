"""Spec section 9, one test per required case, in the spec's own order.

These are the gaps the prototype's suite left open. Each test names the rule it
enforces so a failure points straight at the clause it broke. Everything here
runs without Ollama, without ComfyUI, without ffmpeg.
"""

import asyncio
import json

import pytest

from opera.bible import BibleWriter, ProjectStore
from opera.config import LoopConfig, OperaConfig, RoleConfig
from opera.llm.parsing import extract_json
from opera.llm.stub import think_wrapped
from opera.loop import execute_task
from opera.registry import EngineSpec
from opera.runner import Runner
from opera.schemas import RunStatus, Task, TaskStatus, Verdict
from tests.doubles import (
    ScriptedJudge,
    ScriptedProducer,
    StaticPlanner,
    UnavailableProducer,
)

ROLES = {
    "writer": RoleConfig(model="m", kind="text", max_attempts=2),
    "painter": RoleConfig(model="m", kind="image", max_attempts=2),
}


def spec(**over) -> EngineSpec:
    kwargs = dict(
        name="acceptance", roles=ROLES,
        producers={"writer": ScriptedProducer(["a", "b", "c"], name="writer"),
                   "painter": ScriptedProducer(["<png>"], name="painter", kind="image")},
        judge=ScriptedJudge([True] * 10),
        kinds=frozenset({"text", "image"}),
        default_role="writer", default_kind="text",
        router_keywords={"writer": ["write", "scene"], "painter": ["concept art"]},
    )
    kwargs.update(over)
    return EngineSpec(**kwargs)


# 1 ---------------------------------------------------------------------------

async def test_artifact_verdict_pairing():
    """The final stored artifact was judged -- not a successor to it."""
    task = Task(goal="Write it", role="writer", kind="text")
    producer = ScriptedProducer(["first", "second"])
    judge = ScriptedJudge([False, True])

    await execute_task(task, producer, judge, "", config=LoopConfig(max_attempts=2))

    assert task.artifact.content == "second"
    assert judge.seen[-1][1] is task.artifact
    assert task.artifact.verdict.passed is True


# 2 ---------------------------------------------------------------------------

async def test_attempt_accounting():
    """max_attempts=2 produces exactly two artifacts maximum."""
    task = Task(goal="Write it", role="writer", kind="text")
    producer = ScriptedProducer(["1", "2", "3", "4"])

    await execute_task(task, producer, ScriptedJudge([False] * 4), "",
                       config=LoopConfig(max_attempts=2))

    assert producer.calls == 2
    assert len(task.artifacts) == 2


# 3 ---------------------------------------------------------------------------

async def test_planner_emitting_an_unknown_role_falls_back_and_the_run_completes(tmp_path):
    store = ProjectStore(tmp_path)
    planner = StaticPlanner([{"goal": "Write the opening scene", "role": "cinematographer"}])
    report = await Runner(spec(), store, planner=planner).run(
        store.create("P", "acceptance"), "g")

    assert report.status is RunStatus.DONE
    assert report.run.tasks[0].role == "writer"
    assert report.run.tasks[0].corrections


# 4 ---------------------------------------------------------------------------

async def test_producer_raising_mid_run_fails_only_that_task(tmp_path):
    """Remaining tasks still run, and the project is persisted with partials."""
    store = ProjectStore(tmp_path)
    engine = spec(producers={
        "writer": ScriptedProducer(["ok1", "ok2"], name="writer"),
        "painter": ScriptedProducer(name="painter", kind="image",
                                    raises=RuntimeError("gpu fell over")),
    })
    planner = StaticPlanner([
        {"goal": "Write one", "role": "writer"},
        {"goal": "Paint it", "role": "painter", "kind": "image"},
        {"goal": "Write two", "role": "writer"},
    ])
    project = store.create("P", "acceptance")

    report = await Runner(engine, store, planner=planner).run(project, "g")

    assert [t.status for t in report.run.tasks] == [
        TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.DONE]
    reloaded = store.load(project.id)
    assert [t.status for t in reloaded.runs[0].tasks] == [
        TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.DONE]
    assert "gpu fell over" in reloaded.runs[0].tasks[1].error


# 5 ---------------------------------------------------------------------------

async def test_intra_run_continuity(tmp_path):
    """Task 2's brief contains task 1's excerpt."""
    store = ProjectStore(tmp_path)
    writer = ScriptedProducer(["MIRA CLIMBS THE STAIR", "second"], name="writer")
    planner = StaticPlanner([{"goal": "Write one", "role": "writer"},
                             {"goal": "Write two", "role": "writer"}])

    await Runner(spec(producers={"writer": writer,
                                 "painter": ScriptedProducer(name="painter", kind="image")}),
                 store, planner=planner).run(store.create("P", "acceptance"), "g")

    assert "MIRA CLIMBS THE STAIR" in writer.briefs[1].context


# 6 ---------------------------------------------------------------------------

async def test_concurrency_two_projects_neither_file_corrupted(tmp_path):
    store = ProjectStore(tmp_path)
    planner = StaticPlanner([{"goal": f"Write {i}", "role": "writer"} for i in range(4)])

    def engine(tag):
        return spec(producers={
            "writer": ScriptedProducer([f"{tag}{i}" for i in range(5)], name="writer",
                                       delay=0.001),
            "painter": ScriptedProducer(name="painter", kind="image")})

    pa, pb = store.create("Alpha", "acceptance"), store.create("Beta", "acceptance")
    ra, rb = await asyncio.gather(
        Runner(engine("alpha-"), store, planner=planner).run(pa, "alpha"),
        Runner(engine("beta-"), store, planner=planner).run(pb, "beta"),
    )

    assert ra.status is RunStatus.DONE and rb.status is RunStatus.DONE
    da = json.loads(store.path_for(pa.id).read_text())
    db = json.loads(store.path_for(pb.id).read_text())
    assert da["name"] == "Alpha" and db["name"] == "Beta"
    assert "beta-" not in json.dumps(da)
    assert "alpha-" not in json.dumps(db)


# 7 ---------------------------------------------------------------------------

async def test_unavailable_producer_defers_with_no_synthetic_content(tmp_path):
    store = ProjectStore(tmp_path)
    engine = spec(producers={"writer": ScriptedProducer(name="writer"),
                             "painter": UnavailableProducer(name="comfyui")})
    planner = StaticPlanner([{"goal": "Paint it", "role": "painter", "kind": "image"}])
    project = store.create("P", "acceptance")

    report = await Runner(engine, store, planner=planner).run(project, "g")
    task = report.run.tasks[0]

    assert task.status is TaskStatus.DEFERRED
    assert task.deferred_reason
    assert task.artifacts == []
    assert task.verdict is None
    # Nothing synthetic reached project memory either.
    assert not any(e.category == "artifacts" for e in store.load(project.id).bible.entries)


# 8 ---------------------------------------------------------------------------

async def test_mixed_run_reports_the_deferral_rather_than_collapsing_to_done(tmp_path):
    store = ProjectStore(tmp_path)
    engine = spec(producers={"writer": ScriptedProducer(["text"], name="writer"),
                             "painter": UnavailableProducer(name="comfyui")})
    planner = StaticPlanner([{"goal": "Write one", "role": "writer"},
                             {"goal": "Paint it", "role": "painter", "kind": "image"}])

    report = await Runner(engine, store, planner=planner).run(
        store.create("P", "acceptance"), "g")

    assert report.status is not RunStatus.DONE
    assert report.run.status_counts() == {"done": 1, "deferred": 1}


# 9 ---------------------------------------------------------------------------

async def test_judged_field_propagates_to_the_stored_verdict():
    task = Task(goal="Render it", role="painter", kind="image")
    judge = ScriptedJudge(
        [Verdict(score=0.9, passed=True, issues=[], judged="plan", judge_name="composite")])

    await execute_task(task, ScriptedProducer(["x"], kind="image"), judge, "")

    assert task.artifact.verdict.judged == "plan"
    assert task.artifact.verdict.judge_name == "composite"


# 10 --------------------------------------------------------------------------

def test_think_wrapped_json_parses_correctly():
    payload = {"tasks": [{"goal": "Write it", "role": "writer"}]}
    assert extract_json(think_wrapped(payload)) == payload
    # And the brace inside the trace is not what gets parsed.
    assert extract_json('<think>{"score": 0.1}</think>{"score": 0.9}')["score"] == 0.9
