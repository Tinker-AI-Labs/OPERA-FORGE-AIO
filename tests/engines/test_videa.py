"""Step 6 -- VIDEA on the new core, built from the spec's own role vocabulary
(no prototype exists on this float to port from or compare against -- see
the build session notes).

Runs entirely on the stub client; no Ollama, no ffmpeg.
"""

import json

import pytest

from engines import videa
from opera.bible import ProjectStore
from opera.config import LLMConfig, OperaConfig, RoleConfig
from opera.errors import ProducerError, ProducerUnavailable
from opera.llm.stub import StubLLMClient, think_wrapped
from opera.planner import LLMPlanner
from opera.runner import Runner
from opera.schemas import Artifact, Brief, RunStatus, TaskStatus

PASS = '{"score": 0.9, "passed": true, "issues": []}'
FAIL = '{"score": 0.3, "passed": false, "issues": ["too thin"]}'


def client_for(*, judge=PASS, plan=None, work="THE LAMP TURNED.") -> StubLLMClient:
    """Routes on the system prompts each role actually uses."""
    plan = plan or json.dumps({"tasks": [{"goal": "Write scene one", "role": "writer",
                                          "kind": "text"}]})
    return StubLLMClient(
        [
            ("You are the producer on a film project", plan),
            ("You review work for a film production", judge),
        ],
        default=work,
    )


def make(client, tmp_path, *, llm=None, **kw):
    llm = llm or LLMConfig(default_model="test-text", judge_model="test-judge",
                           vision_model="test-vision", planner_model="test-plan")
    spec = videa.build(client, llm=llm, with_video_judge=False, **kw)
    store = ProjectStore(tmp_path)
    planner = LLMPlanner(client, spec, model=llm.planner_model)
    return Runner(spec, store, planner=planner, config=OperaConfig()), store, spec


# --- the spec itself ---------------------------------------------------------

def test_videa_declares_its_own_vocabulary():
    spec = videa.build(StubLLMClient())
    assert set(spec.roles) == {"writer", "reasoner", "coder"}
    assert spec.kinds == {"text", "code", "video"}
    assert spec.default_role == "writer"


def test_videa_is_registered():
    from opera import registry

    assert "videa" in registry.available()
    assert registry.get("videa", client=StubLLMClient()).name == "videa"


def test_models_come_from_config_not_from_code():
    spec = videa.build(StubLLMClient(), llm=LLMConfig(default_model="custom:70b"))
    assert all(r.model == "custom:70b" for r in spec.roles.values())


def test_role_configs_are_overridable():
    roles = {"narrator": RoleConfig(model="m", kind="text")}
    spec = videa.build(StubLLMClient(), roles=roles, )
    assert set(spec.roles) == {"narrator"}


def test_videa_routing_uses_its_own_keywords():
    router = videa.build(StubLLMClient()).router()
    assert router.route("write the opening scene").role == "writer"
    assert router.route("refactor the ffmpeg pipeline").role == "coder"
    assert router.route("analyse the pacing of act two").role == "reasoner"


# --- the LLM producer --------------------------------------------------------

async def test_producer_makes_a_text_artifact():
    client = StubLLMClient(default="A lighthouse at dusk.")
    cfg = videa.default_roles()["writer"]
    art = await videa.LLMProducer(client, "writer", cfg).produce(
        Brief(task_id="t1", goal="Write it", kind="text", role="writer"))
    assert art.content == "A lighthouse at dusk."
    assert art.kind == "text" and art.producer == "writer"
    assert art.meta["model"] == cfg.model


async def test_revision_prompt_carries_the_prior_work_and_every_issue():
    client = StubLLMClient(default="revised")
    cfg = videa.default_roles()["writer"]
    brief = Brief(
        task_id="t1", goal="Write it", kind="text", role="writer", attempt=2,
        prior=Artifact(kind="text", content="the first draft"),
        issues=["too short", "wrong tone"],
    )
    await videa.LLMProducer(client, "writer", cfg).produce(brief)
    prompt = client.calls[0]["prompt"]
    assert "the first draft" in prompt
    assert "too short" in prompt and "wrong tone" in prompt
    assert "full revised work" in prompt


async def test_first_pass_prompt_has_no_revision_scaffolding():
    client = StubLLMClient(default="draft")
    cfg = videa.default_roles()["writer"]
    await videa.LLMProducer(client, "writer", cfg).produce(
        Brief(task_id="t1", goal="Write it", kind="text", role="writer"))
    assert "previous attempt" not in client.calls[0]["prompt"]


async def test_producer_requires_an_explicit_client():
    with pytest.raises(ProducerError):
        videa.LLMProducer(None, "writer", videa.default_roles()["writer"])


async def test_empty_completion_is_an_error_not_an_empty_artifact():
    cfg = videa.default_roles()["writer"]
    with pytest.raises(ProducerError):
        await videa.LLMProducer(StubLLMClient(default="   "), "writer", cfg).produce(
            Brief(task_id="t1", goal="g", kind="text", role="writer"))


async def test_unreachable_host_defers_rather_than_faking():
    class Dead:
        name = "dead"

        async def complete(self, **kw):
            raise ConnectionError("connection refused")

        async def available(self):
            return False

    cfg = videa.default_roles()["writer"]
    with pytest.raises(ProducerUnavailable):
        await videa.LLMProducer(Dead(), "writer", cfg).produce(
            Brief(task_id="t1", goal="g", kind="text", role="writer"))


async def test_reasoner_keeps_its_reasoning_traces():
    """The one role where /no_think would throw away the product."""
    assert videa.default_roles()["reasoner"].no_think is False
    assert videa.default_roles()["writer"].no_think is True


# --- end to end on the stub --------------------------------------------------

async def test_single_task_run_completes(tmp_path):
    runner, store, _ = make(client_for(), tmp_path)
    report = await runner.run(store.create("Film", "videa"), "write the opening")
    assert report.status is RunStatus.DONE
    assert report.run.tasks[0].artifact.content == "THE LAMP TURNED."
    assert report.run.tasks[0].artifact.verdict.judged == "artifact"


async def test_revision_loop_runs_end_to_end(tmp_path):
    verdicts = iter([FAIL, PASS])
    client = StubLLMClient(
        [
            ("You are the producer on a film project",
             json.dumps({"tasks": [{"goal": "Write scene one", "role": "writer"}]})),
            ("You review work for a film production", lambda s, p: next(verdicts)),
        ],
        default="draft text",
    )
    runner, store, _ = make(client, tmp_path)
    report = await runner.run(store.create("Film", "videa"), "write it")
    task = report.run.tasks[0]
    assert task.attempts == 2
    assert task.status is TaskStatus.DONE
    assert task.artifact.verdict.passed is True   # the revision was judged


async def test_multi_scene_run_carries_continuity(tmp_path):
    plan = json.dumps({"tasks": [
        {"goal": "Write scene one", "role": "writer", "kind": "text"},
        {"goal": "Write scene two", "role": "writer", "kind": "text"},
    ]})
    work = iter(["MIRA CLIMBS THE STAIR.", "MIRA REACHES THE LAMP."])
    client = StubLLMClient(
        [("You are the producer on a film project", plan),
         ("You review work for a film production", PASS)],
        default=lambda s, p: next(work),
    )
    runner, store, _ = make(client, tmp_path)
    report = await runner.run(store.create("Film", "videa"), "two scenes")

    assert report.status is RunStatus.DONE
    # The second writer call must have seen the first scene's text.
    writer_calls = [c for c in client.calls if "screenwriter" in c["system"]]
    assert "MIRA CLIMBS THE STAIR." in writer_calls[1]["prompt"]


async def test_planner_hallucination_does_not_sink_the_run(tmp_path):
    plan = json.dumps({"tasks": [
        {"goal": "Write the opening scene", "role": "cinematographer", "kind": "hologram"},
        {"goal": "Refactor the ffmpeg pipeline", "role": "gaffer", "kind": "code"},
    ]})
    runner, store, _ = make(client_for(plan=plan), tmp_path)
    report = await runner.run(store.create("Film", "videa"), "make a film")

    assert report.status is RunStatus.DONE
    assert [t.role for t in report.run.tasks] == ["writer", "coder"]
    assert all(t.corrections for t in report.run.tasks)


async def test_think_wrapped_planner_and_judge_work_end_to_end(tmp_path):
    """qwen3 defaults, end to end."""
    plan = think_wrapped({"tasks": [{"goal": "Write scene one", "role": "writer"}]})
    client = client_for(plan=plan, judge=think_wrapped(json.loads(PASS)))
    runner, store, _ = make(client, tmp_path)
    report = await runner.run(store.create("Film", "videa"), "write it")
    assert report.status is RunStatus.DONE


async def test_project_memory_persists_between_runs(tmp_path):
    runner, store, _ = make(client_for(), tmp_path)
    project = store.create("Film", "videa")
    await runner.run(project, "write the opening")

    reloaded = store.load(project.id)
    categories = {e.category for e in reloaded.bible.entries}
    assert "artifacts" in categories and "brief" in categories
    assert any(e.event == "task_complete" for e in reloaded.ledger.entries)


async def test_second_run_prompts_contain_no_ledger_strings(tmp_path):
    client = client_for()
    runner, store, _ = make(client, tmp_path)
    project = store.create("Film", "videa")
    await runner.run(project, "write the opening")
    await runner.run(project, "write another")

    for call in client.calls:
        blob = call["system"] + call["prompt"]
        for leak in ("attempts=", "status=done", "task_complete", "duration_s"):
            assert leak not in blob


async def test_video_judge_is_optional_and_reports_frames():
    spec = videa.build(StubLLMClient(), with_video_judge=True)
    assert spec.judge.video_judge is not None
    assert spec.judge.video_judge.name == "videa-frames"
