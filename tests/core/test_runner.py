"""Spec 4.3 / 4.4 / 4.5 / 4.6 / 4.7 -- the run-scoped rules."""

import asyncio
import json

import pytest

from opera.bible import BibleWriter, ProjectStore
from opera.config import LoopConfig, OperaConfig, RoleConfig
from opera.errors import ProducerUnavailable
from opera.planner import Plan, PlannedTask
from opera.registry import EngineSpec
from opera.runner import Runner
from opera.schemas import RunStatus, TaskStatus
from tests.doubles import (
    CallablePlanner,
    ScriptedJudge,
    ScriptedProducer,
    StaticPlanner,
    UnavailableProducer,
)

ROLES = {
    "writer": RoleConfig(model="test-model", kind="text", max_attempts=2),
    "coder": RoleConfig(model="test-model", kind="code", max_attempts=2),
    "painter": RoleConfig(model="test-vision", kind="image", max_attempts=2),
}


def make_spec(**over) -> EngineSpec:
    producers = over.pop("producers", None) or {
        "writer": ScriptedProducer(["scene one text", "scene two text", "scene three text"],
                                   name="writer"),
        "coder": ScriptedProducer(["print('hi')"], name="coder", kind="code"),
        "painter": ScriptedProducer(["<png>"], name="painter", kind="image"),
    }
    kwargs = dict(
        name="testengine",
        roles=ROLES,
        producers=producers,
        judge=ScriptedJudge([True] * 20),
        kinds=frozenset({"text", "code", "image"}),
        default_role="writer",
        default_kind="text",
        router_keywords={
            "writer": ["write", "script", "scene", "dialogue"],
            "coder": ["code", "function", "refactor"],
            "painter": ["concept art", "key frame", "illustration"],
        },
    )
    kwargs.update(over)
    return EngineSpec(**kwargs)


def make_runner(tmp_path, spec=None, planner=None, config=None) -> tuple[Runner, ProjectStore]:
    store = ProjectStore(tmp_path)
    spec = spec or make_spec()
    runner = Runner(spec, store, planner=planner, config=config or OperaConfig())
    return runner, store


# --- 4.3 context is recomputed per task --------------------------------------

async def test_task_two_sees_task_one_output():
    """Intra-run continuity. A per-run snapshot would make this impossible."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        planner = StaticPlanner([
            {"goal": "Write scene one", "role": "writer", "kind": "text"},
            {"goal": "Write scene two", "role": "writer", "kind": "text"},
        ])
        writer = ScriptedProducer(["THE LAMP TURNED SLOWLY", "scene two"], name="writer")
        spec = make_spec(producers={"writer": writer,
                                    "coder": ScriptedProducer(name="coder", kind="code"),
                                    "painter": ScriptedProducer(name="painter", kind="image")})
        runner, store = make_runner(td, spec=spec, planner=planner)
        project = store.create("P", "testengine")

        await runner.run(project, "two scenes")

        second_brief = writer.briefs[1]
        assert "THE LAMP TURNED SLOWLY" in second_brief.context
        assert "Write scene one" in second_brief.context


async def test_first_task_context_contains_the_pinned_goal(tmp_path):
    writer = ScriptedProducer(["x"], name="writer")
    spec = make_spec(producers={"writer": writer,
                                "coder": ScriptedProducer(name="coder", kind="code"),
                                "painter": ScriptedProducer(name="painter", kind="image")})
    runner, store = make_runner(tmp_path, spec=spec,
                                planner=StaticPlanner([{"goal": "Write it", "role": "writer"}]))
    await runner.run(store.create("P", "testengine"), "A film about a lighthouse")
    assert "lighthouse" in writer.briefs[0].context


async def test_failed_task_output_does_not_enter_the_bible(tmp_path):
    spec = make_spec(judge=ScriptedJudge([False] * 5))
    runner, store = make_runner(tmp_path, spec=spec,
                                planner=StaticPlanner([{"goal": "Write it", "role": "writer"}]))
    project = store.create("P", "testengine")
    await runner.run(project, "goal")
    assert not any(e.category == "artifacts" for e in project.bible.entries)


# --- 4.4 a failing task fails that task, not the run -------------------------

async def test_producer_crash_fails_one_task_and_the_rest_still_run(tmp_path):
    boom = ScriptedProducer(name="coder", kind="code", raises=RuntimeError("compiler exploded"))
    spec = make_spec(producers={
        "writer": ScriptedProducer(["a", "b"], name="writer"),
        "coder": boom,
        "painter": ScriptedProducer(["<png>"], name="painter", kind="image"),
    })
    planner = StaticPlanner([
        {"goal": "Write scene one", "role": "writer"},
        {"goal": "Write the code", "role": "coder", "kind": "code"},
        {"goal": "Write scene two", "role": "writer"},
    ])
    runner, store = make_runner(tmp_path, spec=spec, planner=planner)
    project = store.create("P", "testengine")

    report = await runner.run(project, "mixed")

    statuses = [t.status for t in report.run.tasks]
    assert statuses == [TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.DONE]
    assert "compiler exploded" in report.run.tasks[1].error
    assert report.run.status is RunStatus.PARTIAL

    # 4.6: partial results are on disk, not just in memory.
    reloaded = store.load(project.id)
    assert [t.status for t in reloaded.runs[0].tasks] == statuses


async def test_four_completed_scenes_survive_a_fifth_task_crash(tmp_path):
    boom = ScriptedProducer(name="painter", kind="image", raises=RuntimeError("no gpu"))
    spec = make_spec(producers={
        "writer": ScriptedProducer([f"scene {i}" for i in range(6)], name="writer"),
        "coder": ScriptedProducer(name="coder", kind="code"),
        "painter": boom,
    })
    planner = StaticPlanner(
        [{"goal": f"Write scene {i}", "role": "writer"} for i in range(4)]
        + [{"goal": "Paint the poster", "role": "painter", "kind": "image"}]
    )
    runner, store = make_runner(tmp_path, spec=spec, planner=planner)
    report = await runner.run(store.create("P", "testengine"), "five things")
    assert report.tasks_done == 4
    assert report.tasks_failed == 1


async def test_judge_crash_fails_only_that_task(tmp_path):
    from tests.doubles import ExplodingJudge

    spec = make_spec(judge=ExplodingJudge())
    runner, store = make_runner(tmp_path, spec=spec, planner=StaticPlanner(
        [{"goal": "Write one", "role": "writer"}, {"goal": "Write two", "role": "writer"}]))
    report = await runner.run(store.create("P", "testengine"), "g")
    assert report.tasks_failed == 2
    assert all("JudgeError" in t.error for t in report.run.tasks)


# --- 4.5 planner output is validated -----------------------------------------

async def test_hallucinated_role_falls_back_to_the_router(tmp_path):
    """The stub planner in the prototype only ever emitted valid roles, so this
    path could never be exercised. Feed it garbage on purpose."""
    planner = StaticPlanner([
        {"goal": "Write the opening scene", "role": "cinematographer", "kind": "text"},
        {"goal": "Refactor the render function", "role": "gaffer", "kind": "code"},
    ])
    runner, store = make_runner(tmp_path, planner=planner)
    report = await runner.run(store.create("P", "testengine"), "g")

    assert report.run.status is RunStatus.DONE
    assert report.run.tasks[0].role == "writer"     # routed on "write"/"scene"
    assert report.run.tasks[1].role == "coder"      # routed on "refactor"
    assert "cinematographer" in report.run.tasks[0].corrections[0]


async def test_unknown_kind_falls_back_to_the_engine_default(tmp_path):
    planner = StaticPlanner([{"goal": "Write it", "role": "writer", "kind": "hologram"}])
    runner, store = make_runner(tmp_path, planner=planner)
    report = await runner.run(store.create("P", "testengine"), "g")
    task = report.run.tasks[0]
    assert task.kind == "text"
    assert any("hologram" in c for c in task.corrections)


async def test_valid_role_with_mismatched_kind_is_corrected(tmp_path):
    planner = StaticPlanner([{"goal": "Write it", "role": "writer", "kind": "image"}])
    runner, store = make_runner(tmp_path, planner=planner)
    task = (await runner.run(store.create("P", "testengine"), "g")).run.tasks[0]
    assert task.kind == "text"
    assert any("not produced by role" in c for c in task.corrections)


async def test_missing_role_is_routed(tmp_path):
    planner = StaticPlanner([{"goal": "Refactor the parser function"}])
    runner, store = make_runner(tmp_path, planner=planner)
    task = (await runner.run(store.create("P", "testengine"), "g")).run.tasks[0]
    assert task.role == "coder"


async def test_corrections_are_logged_to_the_ledger(tmp_path):
    planner = StaticPlanner([{"goal": "Write it", "role": "nonsense"}])
    runner, store = make_runner(tmp_path, planner=planner)
    project = store.create("P", "testengine")
    await runner.run(project, "g")
    events = [e.event for e in project.ledger.entries]
    assert "plan_corrected" in events


async def test_planner_returning_nothing_fails_the_run_cleanly(tmp_path):
    runner, store = make_runner(tmp_path, planner=StaticPlanner([]))
    project = store.create("P", "testengine")
    report = await runner.run(project, "g")
    assert report.status is RunStatus.FAILED
    assert "planning failed" in report.run.error
    assert store.load(project.id).runs[0].status is RunStatus.FAILED


# --- 4.6 incremental persistence ---------------------------------------------

async def test_project_is_saved_after_every_task(tmp_path):
    saves = []
    runner, store = make_runner(tmp_path, planner=StaticPlanner(
        [{"goal": f"Write {i}", "role": "writer"} for i in range(3)]))
    project = store.create("P", "testengine")
    original = store.save

    def counting_save(p):
        saves.append(len([t for r in p.runs for t in r.tasks
                          if t.status is not TaskStatus.PENDING]))
        return original(p)

    store.save = counting_save
    await runner.run(project, "g")
    # The project reaches disk with 1, then 2, then 3 completed tasks -- never
    # only once at the end. Extra saves are fine; skipping a milestone is not.
    milestones = []
    for n in saves:
        if not milestones or n != milestones[-1]:
            milestones.append(n)
    assert milestones == [0, 1, 2, 3]


async def test_crash_after_task_two_leaves_two_tasks_on_disk(tmp_path):
    calls = {"n": 0}

    class BlowsUpOnThird:
        name, kind, available = "writer", "text", True

        async def produce(self, brief):
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt("power cut")
            from opera.schemas import Artifact
            return Artifact(kind="text", content=f"scene {calls['n']}", producer="writer")

    spec = make_spec(producers={"writer": BlowsUpOnThird(),
                                "coder": ScriptedProducer(name="coder", kind="code"),
                                "painter": ScriptedProducer(name="painter", kind="image")})
    runner, store = make_runner(tmp_path, spec=spec, planner=StaticPlanner(
        [{"goal": f"Write {i}", "role": "writer"} for i in range(4)]))
    project = store.create("P", "testengine")

    with pytest.raises(KeyboardInterrupt):
        await runner.run(project, "g")

    reloaded = store.load(project.id)
    done = [t for t in reloaded.runs[0].tasks if t.status is TaskStatus.DONE]
    assert len(done) == 2
    assert done[0].artifact.content == "scene 1"


# --- 4.7 no shared mutable run state -----------------------------------------

async def test_two_projects_run_concurrently_without_corrupting_either(tmp_path):
    store = ProjectStore(tmp_path)
    spec_a = make_spec(producers={
        "writer": ScriptedProducer([f"alpha {i}" for i in range(6)], name="writer", delay=0.002),
        "coder": ScriptedProducer(name="coder", kind="code"),
        "painter": ScriptedProducer(name="painter", kind="image")})
    spec_b = make_spec(producers={
        "writer": ScriptedProducer([f"beta {i}" for i in range(6)], name="writer", delay=0.002),
        "coder": ScriptedProducer(name="coder", kind="code"),
        "painter": ScriptedProducer(name="painter", kind="image")})
    planner = StaticPlanner([{"goal": f"Write {i}", "role": "writer"} for i in range(5)])

    runner_a = Runner(spec_a, store, planner=planner)
    runner_b = Runner(spec_b, store, planner=planner)
    pa = store.create("Alpha", "testengine")
    pb = store.create("Beta", "testengine")

    ra, rb = await asyncio.gather(runner_a.run(pa, "alpha goal"), runner_b.run(pb, "beta goal"))

    assert ra.status is RunStatus.DONE and rb.status is RunStatus.DONE

    la, lb = store.load(pa.id), store.load(pb.id)
    assert la.name == "Alpha" and lb.name == "Beta"
    assert len(la.runs[0].tasks) == 5 and len(lb.runs[0].tasks) == 5
    # No cross-contamination of content between the two projects' files.
    text_a = json.dumps(json.loads(store.path_for(pa.id).read_text()))
    text_b = json.dumps(json.loads(store.path_for(pb.id).read_text()))
    assert "beta " not in text_a
    assert "alpha " not in text_b


async def test_one_runner_serving_two_projects_keeps_them_separate(tmp_path):
    runner, store = make_runner(tmp_path, planner=StaticPlanner(
        [{"goal": "Write it", "role": "writer"}]))
    pa, pb = store.create("A", "testengine"), store.create("B", "testengine")
    await asyncio.gather(runner.run(pa, "goal a"), runner.run(pb, "goal b"))
    la, lb = store.load(pa.id), store.load(pb.id)
    assert len(la.runs) == 1 and len(lb.runs) == 1
    assert la.runs[0].goal == "goal a" and lb.runs[0].goal == "goal b"
    assert "goal b" not in json.dumps(la.model_dump(mode="json"))


# --- aggregate status honesty ------------------------------------------------

async def test_mixed_done_and_deferred_does_not_collapse_to_done(tmp_path):
    spec = make_spec(producers={
        "writer": ScriptedProducer(["scene"], name="writer"),
        "coder": ScriptedProducer(name="coder", kind="code"),
        "painter": UnavailableProducer(name="comfyui", kind="image")})
    planner = StaticPlanner([
        {"goal": "Write scene one", "role": "writer"},
        {"goal": "Paint the key frame", "role": "painter", "kind": "image"},
    ])
    runner, store = make_runner(tmp_path, spec=spec, planner=planner)
    report = await runner.run(store.create("P", "testengine"), "g")

    assert report.status is not RunStatus.DONE
    assert report.status is RunStatus.PARTIAL
    assert report.run.status_counts() == {"done": 1, "deferred": 1}
    deferred = report.run.tasks[1]
    assert deferred.deferred_reason and deferred.artifacts == []


async def test_all_deferred_reports_deferred(tmp_path):
    spec = make_spec(producers={
        "writer": UnavailableProducer(name="offline-writer", kind="text"),
        "coder": ScriptedProducer(name="coder", kind="code"),
        "painter": ScriptedProducer(name="painter", kind="image")})
    runner, store = make_runner(tmp_path, spec=spec, planner=StaticPlanner(
        [{"goal": "Write it", "role": "writer"}]))
    report = await runner.run(store.create("P", "testengine"), "g")
    assert report.status is RunStatus.DEFERRED


async def test_deferred_result_contains_no_synthetic_content(tmp_path):
    spec = make_spec(producers={
        "writer": ScriptedProducer(["real text"], name="writer"),
        "coder": ScriptedProducer(name="coder", kind="code"),
        "painter": UnavailableProducer(name="comfyui", kind="image")})
    runner, store = make_runner(tmp_path, spec=spec, planner=StaticPlanner(
        [{"goal": "Paint it", "role": "painter", "kind": "image"}]))
    project = store.create("P", "testengine")
    report = await runner.run(project, "g")
    task = report.run.tasks[0]
    assert task.artifacts == [] and task.verdict is None
    assert not any(e.category == "artifacts" for e in project.bible.entries)


async def test_ledger_records_telemetry_and_bible_does_not(tmp_path):
    runner, store = make_runner(tmp_path, planner=StaticPlanner(
        [{"goal": "Write it", "role": "writer"}]))
    project = store.create("P", "testengine")
    await runner.run(project, "g")

    entry = [e for e in project.ledger.entries if e.event == "task_complete"][0]
    assert entry.status == "done" and entry.attempts == 1 and entry.model == "test-model"

    context = BibleWriter().context(project.bible)
    for leak in ("attempts", "status=", "task_complete", "test-model"):
        assert leak not in context


async def test_planner_receives_existing_project_context(tmp_path):
    seen = {}

    def fn(goal, context):
        seen["context"] = context
        return Plan(tasks=[PlannedTask(goal="Write it", role="writer")])

    runner, store = make_runner(tmp_path, planner=CallablePlanner(fn))
    project = store.create("P", "testengine")
    BibleWriter().add(project.bible, "characters", "Mira keeps the light.")
    await runner.run(project, "g")
    assert "Mira keeps the light." in seen["context"]


async def test_run_project_loads_by_id(tmp_path):
    runner, store = make_runner(tmp_path, planner=StaticPlanner(
        [{"goal": "Write it", "role": "writer"}]))
    project = store.create("P", "testengine")
    report = await runner.run_project(project.id, "g")
    assert report.status is RunStatus.DONE


# --- rerun_task: the run dashboard's data layer -------------------------

async def test_rerun_task_adds_a_new_attempt_without_discarding_history(tmp_path):
    """A dashboard's "re-run this task" action: a failed task gets exactly
    one more production, appended -- not a fresh task, not a lost attempt."""
    producer = ScriptedProducer(["bad first draft", "much better draft"], name="writer")
    judge = ScriptedJudge([False, True])
    spec = make_spec(
        roles={"writer": RoleConfig(model="test-model", kind="text", max_attempts=1)},
        producers={"writer": producer},
        judge=judge,
    )
    runner, store = make_runner(
        tmp_path, spec=spec, planner=StaticPlanner([{"goal": "Write it", "role": "writer"}]))
    project = store.create("P", "testengine")

    report = await runner.run(project, "g")
    task = report.run.tasks[0]
    assert task.status is TaskStatus.FAILED
    assert task.attempts == 1
    assert [a.content for a in task.artifacts] == ["bad first draft"]

    rerun_report = await runner.rerun_task(project, report.run.id, task.id)
    rerun_task = rerun_report.run.tasks[0]

    assert rerun_task.status is TaskStatus.DONE
    assert rerun_task.attempts == 2
    assert [a.content for a in rerun_task.artifacts] == ["bad first draft", "much better draft"]
    assert rerun_task.error is None, "a stale FAILED-run error must not survive a passing rerun"

    # 4.6: the rerun is persisted immediately, same discipline as a normal run.
    reloaded = store.load(project.id).runs[0].tasks[0]
    assert reloaded.status is TaskStatus.DONE
    assert len(reloaded.artifacts) == 2


async def test_rerun_task_only_touches_the_named_task(tmp_path):
    spec = make_spec()
    runner, store = make_runner(tmp_path, spec=spec, planner=StaticPlanner([
        {"goal": "Write scene one", "role": "writer"},
        {"goal": "Write scene two", "role": "writer"},
    ]))
    project = store.create("P", "testengine")
    report = await runner.run(project, "g")
    other = report.run.tasks[1]
    other_attempts_before = other.attempts
    other_artifacts_before = list(other.artifacts)

    target = report.run.tasks[0]
    rerun_report = await runner.rerun_task(project, report.run.id, target.id)

    untouched = rerun_report.run.tasks[1]
    assert untouched.attempts == other_attempts_before
    assert untouched.artifacts == other_artifacts_before


async def test_rerun_task_unknown_run_id_raises(tmp_path):
    runner, store = make_runner(tmp_path)
    project = store.create("P", "testengine")
    with pytest.raises(ValueError, match="run"):
        await runner.rerun_task(project, "run_nope", "t_nope")


async def test_rerun_task_unknown_task_id_raises(tmp_path):
    runner, store = make_runner(
        tmp_path, planner=StaticPlanner([{"goal": "Write it", "role": "writer"}]))
    project = store.create("P", "testengine")
    report = await runner.run(project, "g")
    with pytest.raises(ValueError, match="task"):
        await runner.rerun_task(project, report.run.id, "t_nope")


async def test_rerun_task_still_defers_when_producer_still_unavailable(tmp_path):
    spec = make_spec(producers={
        "writer": ScriptedProducer(["a"], name="writer"),
        "coder": ScriptedProducer(["b"], name="coder", kind="code"),
        "painter": UnavailableProducer(name="painter", kind="image"),
    })
    runner, store = make_runner(tmp_path, spec=spec, planner=StaticPlanner(
        [{"goal": "Paint it", "role": "painter", "kind": "image"}]))
    project = store.create("P", "testengine")

    report = await runner.run(project, "g")
    task = report.run.tasks[0]
    assert task.status is TaskStatus.DEFERRED

    rerun_report = await runner.rerun_task(project, report.run.id, task.id)
    rerun_task = rerun_report.run.tasks[0]
    assert rerun_task.status is TaskStatus.DEFERRED, (
        "a rerun must not fabricate success just because it was asked for"
    )
    assert rerun_task.artifacts == []


async def test_rerun_task_project_loads_by_id(tmp_path):
    runner, store = make_runner(
        tmp_path, planner=StaticPlanner([{"goal": "Write it", "role": "writer"}]))
    project = store.create("P", "testengine")
    report = await runner.run_project(project.id, "g")
    task = report.run.tasks[0]

    rerun_report = await runner.rerun_task_project(project.id, report.run.id, task.id)
    assert rerun_report.run.tasks[0].attempts == task.attempts + 1
