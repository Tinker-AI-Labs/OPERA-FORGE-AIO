"""Step 7 -- the first real non-LLM producer.

The load-bearing assertion of this whole file is negative: nothing in
``opera/`` had to change to make an image generator work.
"""

import base64
import json

import httpx
import pytest

from engines import artista
from opera.bible import ProjectStore
from opera.config import LLMConfig, OperaConfig, RoleConfig
from opera.errors import ProducerUnavailable
from opera.llm.stub import StubLLMClient
from opera.planner import LLMPlanner
from opera.runner import Runner
from opera.schemas import Artifact, Brief, RunStatus, TaskStatus

PNG = b"\x89PNG\r\n\x1a\nfake image bytes"
WORKFLOW = {"3": {"inputs": {"seed": 0}}, "6": {"inputs": {"text": ""}}}


def comfy_transport(*, fail=None, image=PNG):
    """A minimal ComfyUI stand-in: /prompt, /history/<id>, /view."""
    state = {"submitted": []}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/prompt":
            body = json.loads(request.content)
            state["submitted"].append(body)
            if fail == "submit":
                return httpx.Response(500, text="queue full")
            return httpx.Response(200, json={"prompt_id": "abc123"})
        if path.startswith("/history/"):
            if fail == "job":
                return httpx.Response(200, json={"abc123": {
                    "status": {"status_str": "error", "messages": ["bad model"]}}})
            return httpx.Response(200, json={"abc123": {"outputs": {"9": {"images": [
                {"filename": "out_0001.png", "subfolder": "", "type": "output"}]}}}})
        if path == "/view":
            return httpx.Response(200, content=image)
        return httpx.Response(404)

    return handler, state


def comfy_producer(tmp_path=None, **kw):
    handler, state = comfy_transport(**kw)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://comfy")
    p = artista.ComfyUIProducer(
        "retoucher", RoleConfig(model="flux", kind="image", timeout_s=5.0),
        workflow=WORKFLOW, available=True, client=http, poll_interval_s=0.0,
        output_dir=tmp_path,
    )
    return p, state


# --- the Producer protocol holds for a media generator -----------------------

async def test_comfyui_producer_returns_a_real_image_artifact(tmp_path):
    producer, state = comfy_producer(tmp_path)
    brief = Brief(task_id="t1", goal="a red lighthouse at dusk", kind="image", role="retoucher")

    art = await producer.produce(brief)

    assert art.kind == "image"
    assert art.path and open(art.path, "rb").read() == PNG
    # The bytes live on disk, not inlined in the project JSON.
    assert "image_b64" not in art.meta
    assert art.meta["prompt"] == "a red lighthouse at dusk"
    # The prompt actually reached the workflow node.
    assert state["submitted"][0]["prompt"]["6"]["inputs"]["text"] == "a red lighthouse at dusk"


async def test_producer_satisfies_the_protocol_structurally():
    from opera.protocols import Producer

    producer, _ = comfy_producer()
    assert isinstance(producer, Producer)


async def test_upstream_prompt_artifact_is_used_when_supplied(tmp_path):
    producer, state = comfy_producer(tmp_path)
    brief = Brief(task_id="t1", goal="make it", kind="image", role="retoucher",
                  params={"prompt": "cinematic wide shot, cold blue palette"})
    await producer.produce(brief)
    assert state["submitted"][0]["prompt"]["6"]["inputs"]["text"] == (
        "cinematic wide shot, cold blue palette")


async def test_revision_issues_reach_the_generation_prompt(tmp_path):
    producer, state = comfy_producer(tmp_path)
    brief = Brief(task_id="t1", goal="a lighthouse", kind="image", role="retoucher",
                  attempt=2, prior=Artifact(kind="image"), issues=["tower should be red"])
    await producer.produce(brief)
    assert "tower should be red" in state["submitted"][0]["prompt"]["6"]["inputs"]["text"]


async def test_each_production_gets_a_fresh_seed(tmp_path):
    producer, state = comfy_producer(tmp_path)
    brief = Brief(task_id="t1", goal="x", kind="image", role="retoucher")
    await producer.produce(brief)
    await producer.produce(brief)
    seeds = [s["prompt"]["3"]["inputs"]["seed"] for s in state["submitted"]]
    assert seeds[0] != seeds[1]


# --- unavailability is never faked -------------------------------------------

async def test_image_bytes_are_inlined_only_when_there_is_no_file():
    """Without an output dir there is no path, so the judge still gets the bytes."""
    producer, _ = comfy_producer(tmp_path=None)
    art = await producer.produce(Brief(task_id="t1", goal="x", kind="image", role="retoucher"))
    assert art.path is None
    assert base64.b64decode(art.meta["image_b64"]) == PNG


def test_no_workflow_means_unavailable():
    p = artista.ComfyUIProducer("retoucher", RoleConfig(model="flux", kind="image"))
    assert p.available is False


async def test_comfyui_refusing_the_job_defers_rather_than_faking():
    producer, _ = comfy_producer(fail="submit")
    with pytest.raises(ProducerUnavailable):
        await producer.produce(Brief(task_id="t1", goal="x", kind="image", role="retoucher"))


async def test_comfyui_job_error_is_a_producer_error_not_a_blank_image():
    from opera.errors import ProducerError

    producer, _ = comfy_producer(fail="job")
    with pytest.raises(ProducerError, match="failed"):
        await producer.produce(Brief(task_id="t1", goal="x", kind="image", role="retoucher"))


async def test_unavailable_comfyui_defers_the_task_in_a_real_run(tmp_path):
    """The end-to-end version: no ComfyUI, no fake picture, honest status."""
    client = StubLLMClient(
        [("You are an art director",
          json.dumps({"tasks": [{"goal": "Generate the key art", "role": "retoucher",
                                 "kind": "image"}]}))],
        default='{"score":0.9,"passed":true,"issues":[]}',
    )
    spec = artista.build(client, workflow=None)   # no workflow -> unavailable
    store = ProjectStore(tmp_path)
    runner = Runner(spec, store, planner=LLMPlanner(client, spec, model="m"))

    report = await runner.run(store.create("Poster", "artista"), "make key art")

    task = report.run.tasks[0]
    assert task.status is TaskStatus.DEFERRED
    assert report.status is RunStatus.DEFERRED
    assert task.artifacts == [] and task.verdict is None


# --- the judge reports what it actually assessed -----------------------------

async def test_image_verdict_says_artifact_and_prompt_verdict_says_plan():
    client = StubLLMClient(default='{"score":0.9,"passed":true,"issues":[]}')
    judge = artista.build_judge(client, vision_model="v", text_model="t")
    from opera.schemas import Task

    task = Task(goal="g", role="retoucher", kind="image")
    image_v = await judge.evaluate(task, Artifact(kind="image", meta={"image_b64": "QUJD"}), "")
    prompt_v = await judge.evaluate(task, Artifact(kind="text", content="a prompt"), "")

    assert image_v.judged == "artifact"
    assert prompt_v.judged == "plan"     # a prompt review is not an image review


async def test_vision_judge_actually_receives_the_generated_bytes(tmp_path):
    producer, _ = comfy_producer(tmp_path)
    art = await producer.produce(Brief(task_id="t1", goal="x", kind="image", role="retoucher"))

    client = StubLLMClient(default='{"score":0.8,"passed":true,"issues":[]}')
    judge = artista.build_judge(client, vision_model="v", text_model="t")
    from opera.schemas import Task

    await judge.evaluate(Task(goal="g", role="retoucher", kind="image"), art, "")
    assert base64.b64decode(client.calls[0]["images"][0]) == PNG


# --- the whole loop over a media producer ------------------------------------

async def test_full_produce_judge_revise_cycle_over_images(tmp_path):
    """A media producer gets the same review loop an LLM agent does -- which is
    the thing the prototype did not have."""
    verdicts = iter([
        '{"score":0.3,"passed":false,"issues":["the tower should be red"]}',
        '{"score":0.9,"passed":true,"issues":[]}',
    ])
    client = StubLLMClient(default=lambda s, p: next(verdicts))
    producer, state = comfy_producer(tmp_path)
    judge = artista.build_judge(client, vision_model="v", text_model="t")

    from opera.config import LoopConfig
    from opera.loop import execute_task
    from opera.schemas import Task

    task = Task(goal="key art of a lighthouse", role="retoucher", kind="image")
    await execute_task(task, producer, judge, "cold blue palette",
                       config=LoopConfig(max_attempts=2))

    assert task.status is TaskStatus.DONE
    assert task.attempts == 2
    assert len(state["submitted"]) == 2
    # The revision's prompt carried the judge's issue.
    assert "tower should be red" in state["submitted"][1]["prompt"]["6"]["inputs"]["text"]
    assert task.artifact.verdict.passed is True
    assert task.artifact.verdict.judged == "artifact"


async def test_two_stage_run_prompt_then_image(tmp_path):
    """prompt_smith writes text, retoucher makes the picture from the bible."""
    plan = json.dumps({"tasks": [
        {"goal": "Write the image prompt for the poster", "role": "prompt_smith", "kind": "text"},
        {"goal": "Generate the key art", "role": "retoucher", "kind": "image"},
    ]})
    client = StubLLMClient(
        [("You are an art director", plan),
         ("You review image-generation prompts", '{"score":0.9,"passed":true,"issues":[]}'),
         ("You review generated images", '{"score":0.9,"passed":true,"issues":[]}')],
        default="a red lighthouse, cold blue palette, cinematic",
    )
    producer, state = comfy_producer(tmp_path)
    spec = artista.build(client, image_producer=producer)
    store = ProjectStore(tmp_path / "projects")
    runner = Runner(spec, store, planner=LLMPlanner(client, spec, model="m"))

    report = await runner.run(store.create("Poster", "artista"), "a lighthouse poster")

    assert report.status is RunStatus.DONE
    assert [t.kind for t in report.run.tasks] == ["text", "image"]
    # Intra-run continuity across a text -> image boundary: the prompt_smith's
    # words must actually reach the image generator.
    generated_prompt = state["submitted"][0]["prompt"]["6"]["inputs"]["text"]
    assert "a red lighthouse, cold blue palette, cinematic" in generated_prompt
    assert report.run.tasks[0].artifact.verdict.judged == "plan"
    assert report.run.tasks[1].artifact.verdict.judged == "artifact"


# --- vocabulary is ARTISTA's own ---------------------------------------------

def test_artista_vocabulary_is_its_own():
    spec = artista.build(StubLLMClient(), workflow=WORKFLOW)
    assert set(spec.roles) == {"concept", "prompt_smith", "retoucher"}
    assert spec.kinds == {"text", "image"}


def test_artista_routing_does_not_fire_on_state_of_the_art():
    router = artista.build(StubLLMClient(), workflow=WORKFLOW).router()
    assert router.route("summarise the state of the art").role != "retoucher"
    assert router.route("generate the image of the tower").role == "retoucher"
    assert router.route("concept art of the tower").role == "retoucher"


def test_artista_is_registered():
    from opera import registry

    assert "artista" in registry.available()
