"""Spec 3.2 -- the shipped judges, and the honesty of the `judged` field."""

import base64
import json

import pytest

from opera.errors import JudgeError
from opera.judges import (
    CompositeJudge,
    DeterministicJudge,
    FrameSampleJudge,
    LLMJudge,
    VisionJudge,
)
from opera.llm.stub import StubLLMClient, think_wrapped
from opera.schemas import Artifact, Task, Verdict
from tests.doubles import ScriptedJudge

TASK = Task(goal="Write the opening scene", role="writer", kind="text")


def _stub(payload):
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return StubLLMClient(default=body)


# --- LLMJudge ----------------------------------------------------------------

async def test_llm_judge_parses_a_verdict():
    client = _stub({"score": 0.85, "passed": True, "issues": []})
    v = await LLMJudge(client, model="m").evaluate(TASK, Artifact(content="text"), "ctx")
    assert v.score == 0.85 and v.passed is True and v.judged == "artifact"


async def test_llm_judge_handles_think_wrapped_output():
    """The end-to-end version of the parsing bug (spec 5.2)."""
    client = _stub(think_wrapped({"score": 0.4, "passed": False, "issues": ["too thin"]}))
    v = await LLMJudge(client, model="m").evaluate(TASK, Artifact(content="t"), "")
    assert v.passed is False and v.issues == ["too thin"]


async def test_llm_judge_requests_json_and_suppresses_reasoning():
    client = StubLLMClient(default='{"score":1,"passed":true}')
    await LLMJudge(client, model="m").evaluate(TASK, Artifact(content="t"), "")
    call = client.calls[0]
    assert call["format_json"] is True and call["no_think"] is True


async def test_llm_judge_infers_passed_from_threshold_when_absent():
    v = await LLMJudge(_stub({"score": 0.9}), model="m", threshold=0.8).evaluate(
        TASK, Artifact(content="t"), "")
    assert v.passed is True
    v2 = await LLMJudge(_stub({"score": 0.5}), model="m", threshold=0.8).evaluate(
        TASK, Artifact(content="t"), "")
    assert v2.passed is False


async def test_llm_judge_clamps_out_of_range_scores():
    v = await LLMJudge(_stub({"score": 4.2, "passed": True}), model="m").evaluate(
        TASK, Artifact(content="t"), "")
    assert v.score == 1.0


async def test_llm_judge_raises_on_unparseable_output():
    with pytest.raises(JudgeError):
        await LLMJudge(_stub("I think it's pretty good!"), model="m").evaluate(
            TASK, Artifact(content="t"), "")


async def test_llm_judge_raises_on_non_numeric_score():
    with pytest.raises(JudgeError):
        await LLMJudge(_stub({"score": "great"}), model="m").evaluate(
            TASK, Artifact(content="t"), "")


async def test_llm_judge_requires_an_explicit_client():
    with pytest.raises(JudgeError):
        LLMJudge(None, model="m")


async def test_llm_judge_sees_the_context():
    client = StubLLMClient(default='{"score":1,"passed":true}')
    await LLMJudge(client, model="m").evaluate(TASK, Artifact(content="t"), "MIRA IS THE KEEPER")
    assert "MIRA IS THE KEEPER" in client.calls[0]["prompt"]


# --- VisionJudge -------------------------------------------------------------

RUBRIC_GOOD = {
    "subject_present": True,
    "elements": [{"element": "tower", "matched": True}],
    "artifacts_present": False,
    "composition_intact": True,
}


async def test_vision_judge_sends_the_image_as_base64(tmp_path):
    png = tmp_path / "f.png"
    png.write_bytes(b"\x89PNG fake bytes")
    client = _stub(RUBRIC_GOOD)
    art = Artifact(kind="image", path=str(png), meta={"prompt": "a red tower"})

    v = await VisionJudge(client, model="vision").evaluate(TASK, art, "ctx")

    sent = client.calls[0]["images"][0]
    assert base64.b64decode(sent) == b"\x89PNG fake bytes"
    assert "a red tower" in client.calls[0]["prompt"]
    assert v.judged == "artifact"


async def test_vision_judge_accepts_inline_base64():
    client = _stub(RUBRIC_GOOD)
    art = Artifact(kind="image", meta={"image_b64": "QUJD"})
    await VisionJudge(client, model="v").evaluate(TASK, art, "")
    assert client.calls[0]["images"] == ["QUJD"]


async def test_vision_judge_without_an_image_is_an_error():
    with pytest.raises(JudgeError, match="no readable image"):
        await VisionJudge(_stub(RUBRIC_GOOD), model="v").evaluate(TASK, Artifact(kind="image"), "")


async def test_vision_judge_computes_score_from_rubric_not_a_self_reported_number():
    """The fix for the 2026-08-18 finding: llava:7b returned the identical
    0.500 for a plausible-match goal and a wildly wrong one. VisionJudge must
    never ask for or trust a raw number -- the score is computed from the
    rubric's boolean observations."""
    v = await VisionJudge(_stub(RUBRIC_GOOD), model="v").evaluate(
        TASK, Artifact(kind="image", meta={"image_b64": "QUJD"}), "")
    assert v.score == pytest.approx(1.0)
    assert v.passed is True
    assert v.issues == []

    bad = {
        "subject_present": False,
        "subject_note": "no fox visible",
        "elements": [{"element": "snowy forest", "matched": False, "note": "beach instead"}],
        "artifacts_present": True,
        "artifacts_note": "warped limbs",
        "composition_intact": True,
    }
    v2 = await VisionJudge(_stub(bad), model="v").evaluate(
        TASK, Artifact(kind="image", meta={"image_b64": "QUJD"}), "")
    assert v2.score == pytest.approx(0.15)  # composition_intact=True is the only credit earned
    assert v2.passed is False
    assert any("subject not present" in i for i in v2.issues)
    assert any("snowy forest" in i for i in v2.issues)
    assert any("artifacts" in i for i in v2.issues)


async def test_vision_judge_broken_composition_costs_points_but_does_not_zero_a_real_match():
    """2026-08-19 revision, confirmed necessary live against both llava:7b and
    qwen2.5vl:7b: composition_intact must NOT hard-gate the score. Both
    models routinely set it False on a coherent image that simply isn't the
    style/medium they expected (e.g. a 3D render vs. a photo) -- gating on it
    zeroed out otherwise-correct verdicts. It costs its own weight and no
    more, so a real subject/element match still dominates."""
    payload = {
        "subject_present": True,
        "elements": [{"element": "tower", "matched": True}],
        "artifacts_present": False,
        "composition_intact": False,
        "composition_note": "not a photograph, looks like a 3D render",
    }
    v = await VisionJudge(_stub(payload), model="v", threshold=0.7).evaluate(
        TASK, Artifact(kind="image", meta={"image_b64": "QUJD"}), "")
    assert v.score == pytest.approx(0.85)
    assert v.passed is True, "a real subject+element match must survive one wrong composition flag"
    assert "composition is not intact" in v.issues[0]


async def test_vision_judge_never_reads_a_self_reported_score_or_passed_field():
    """Even if a model smuggles score/passed into the JSON (old habit, or a
    model that hasn't been told about the new contract), VisionJudge must
    compute its own verdict from the rubric fields, not read theirs."""
    payload = dict(RUBRIC_GOOD, score=0.01, passed=False)
    v = await VisionJudge(_stub(payload), model="v", threshold=0.5).evaluate(
        TASK, Artifact(kind="image", meta={"image_b64": "QUJD"}), "")
    assert v.score == pytest.approx(1.0)
    assert v.passed is True


@pytest.mark.parametrize("missing_key", [
    "subject_present", "artifacts_present", "composition_intact", "elements",
])
async def test_vision_judge_missing_rubric_field_is_a_failure_not_a_default(missing_key):
    payload = dict(RUBRIC_GOOD)
    del payload[missing_key]
    with pytest.raises(JudgeError):
        await VisionJudge(_stub(payload), model="v").evaluate(
            TASK, Artifact(kind="image", meta={"image_b64": "QUJD"}), "")


async def test_vision_judge_non_boolean_rubric_field_is_a_failure():
    payload = dict(RUBRIC_GOOD, artifacts_present="no")
    with pytest.raises(JudgeError):
        await VisionJudge(_stub(payload), model="v").evaluate(
            TASK, Artifact(kind="image", meta={"image_b64": "QUJD"}), "")


async def test_vision_judge_no_named_elements_defaults_to_no_penalty():
    payload = {
        "subject_present": True,
        "elements": [],
        "artifacts_present": False,
        "composition_intact": True,
    }
    v = await VisionJudge(_stub(payload), model="v").evaluate(
        TASK, Artifact(kind="image", meta={"image_b64": "QUJD"}), "")
    assert v.score == pytest.approx(1.0)


# --- FrameSampleJudge --------------------------------------------------------

async def test_frame_sample_judge_reports_judged_frames(monkeypatch):
    """It never watched the video, and must not imply that it did."""
    from opera.judges import FrameSample

    client = _stub(RUBRIC_GOOD)
    judge = FrameSampleJudge(VisionJudge(client, model="v"), frames=3)
    monkeypatch.setattr(judge, "_extract",
                        lambda path: [FrameSample(index=i, b64="QUJD") for i in range(3)])

    v = await judge.evaluate(TASK, Artifact(kind="video", path="/x.mp4"), "ctx")

    assert v.judged == "frames"
    assert v.detail["frames_sampled"] == 3
    assert "motion, pacing and audio were not assessed" in v.detail["coverage"]
    assert len(client.calls) == 3


async def test_frame_sample_judge_aggregates_and_labels_issues(monkeypatch):
    from opera.judges import FrameSample

    payloads = iter([
        json.dumps(RUBRIC_GOOD),
        json.dumps({
            "subject_present": True,
            "elements": [{"element": "tower", "matched": False, "note": "tower is blue here"}],
            "artifacts_present": False,
            "composition_intact": True,
        }),
    ])
    client = StubLLMClient(default=lambda s, p: next(payloads))
    judge = FrameSampleJudge(VisionJudge(client, model="v"), frames=2)
    monkeypatch.setattr(judge, "_extract",
                        lambda path: [FrameSample(index=i, b64="QUJD") for i in range(2)])

    v = await judge.evaluate(TASK, Artifact(kind="video", path="/x.mp4"), "")

    assert v.score == pytest.approx(0.8)
    assert v.passed is False, "frame 1 scored below its own threshold, so the whole clip must fail"
    assert v.issues == ["frame 1: prompt element not matched: tower (tower is blue here)"]
    assert v.detail["per_frame_scores"] == pytest.approx([1.0, 0.6])


async def test_frame_sample_judge_needs_a_path():
    with pytest.raises(JudgeError, match="no video path"):
        await FrameSampleJudge(VisionJudge(_stub({"score": 1}), model="v")).evaluate(
            TASK, Artifact(kind="video"), "")


def test_frame_sample_judge_reports_ffmpeg_availability():
    judge = FrameSampleJudge(VisionJudge(_stub({}), model="v"), ffmpeg="definitely-not-a-binary")
    assert judge.available() is False


# --- DeterministicJudge ------------------------------------------------------

async def test_deterministic_judge_passes_when_all_checks_pass():
    checks = [("duration", lambda a: (True, "")), ("clipping", lambda a: (True, ""))]
    v = await DeterministicJudge(checks).evaluate(TASK, Artifact(), "")
    assert v.passed is True and v.score == 1.0 and v.issues == []


async def test_deterministic_judge_fails_on_any_failing_check():
    checks = [
        ("duration", lambda a: (True, "")),
        ("key", lambda a: (False, "detected D minor, spec said A minor")),
        ("clipping", lambda a: (True, "")),
    ]
    v = await DeterministicJudge(checks).evaluate(TASK, Artifact(), "")
    assert v.passed is False
    assert v.score == pytest.approx(2 / 3)
    assert v.issues == ["key: detected D minor, spec said A minor"]


async def test_a_raising_check_counts_as_failed_and_says_why():
    def boom(a):
        raise ValueError("no audio stream")

    v = await DeterministicJudge([("decode", boom)]).evaluate(TASK, Artifact(), "")
    assert v.passed is False
    assert "ValueError: no audio stream" in v.issues[0]


async def test_deterministic_judge_with_no_checks_is_vacuously_true():
    v = await DeterministicJudge([]).evaluate(TASK, Artifact(), "")
    assert v.passed is True and v.detail["checks"] == 0


async def test_deterministic_judge_accepts_bare_callables():
    def duration_ok(a):
        return True, ""

    v = await DeterministicJudge([duration_ok]).evaluate(TASK, Artifact(), "")
    assert v.detail["results"] == {"duration_ok": True}


# --- CompositeJudge ----------------------------------------------------------

async def test_composite_all_must_pass():
    j = CompositeJudge([ScriptedJudge([True], name="a"), ScriptedJudge([False], name="b")])
    v = await j.evaluate(TASK, Artifact(), "")
    assert v.passed is False
    assert v.detail["policy"] == "all_must_pass"


async def test_composite_reports_the_union_of_what_was_judged():
    """MUSICA: deterministic checks over the file plus an LLM review of the plan.
    The composite must not claim it assessed the audio's musicality."""
    j = CompositeJudge([
        ScriptedJudge([True], name="checks", judged="artifact"),
        ScriptedJudge([True], name="plan-review", judged="plan"),
    ])
    v = await j.evaluate(TASK, Artifact(), "")
    assert v.judged == "artifact+plan"
    assert set(v.detail["members"]) == {"checks", "plan-review"}


async def test_composite_prefixes_issues_with_their_source():
    j = CompositeJudge([
        ScriptedJudge([False], name="checks"),
        ScriptedJudge([False], name="plan-review"),
    ])
    v = await j.evaluate(TASK, Artifact(), "")
    assert v.issues == ["[checks] not good enough", "[plan-review] not good enough"]


async def test_composite_all_must_pass_takes_the_lowest_score():
    j = CompositeJudge([ScriptedJudge([0.9], name="a"), ScriptedJudge([0.75], name="b")])
    v = await j.evaluate(TASK, Artifact(), "")
    assert v.score == pytest.approx(0.75) and v.passed is True


async def test_composite_any_may_pass():
    j = CompositeJudge([ScriptedJudge([False], name="a"), ScriptedJudge([True], name="b")],
                       policy="any_may_pass")
    v = await j.evaluate(TASK, Artifact(), "")
    assert v.passed is True


async def test_composite_weighted_policy():
    j = CompositeJudge([ScriptedJudge([1.0], name="a"), ScriptedJudge([0.0], name="b")],
                       policy="weighted", weights=[3.0, 1.0], threshold=0.7)
    v = await j.evaluate(TASK, Artifact(), "")
    assert v.score == pytest.approx(0.75) and v.passed is True


async def test_composite_rejects_bad_configuration():
    with pytest.raises(JudgeError):
        CompositeJudge([])
    with pytest.raises(JudgeError):
        CompositeJudge([ScriptedJudge([True])], policy="vibes")
    with pytest.raises(JudgeError):
        CompositeJudge([ScriptedJudge([True])], weights=[1.0, 2.0])


# --- a verdict may not contradict itself -------------------------------------
# Found by a real run against llama3.2:1b, which returned
# {"score": 0.25, "passed": true} against a 0.7 threshold and had the task
# recorded as done.

async def test_a_pass_claim_must_be_backed_by_the_score():
    v = await LLMJudge(_stub({"score": 0.25, "passed": True}), model="m",
                       threshold=0.7).evaluate(TASK, Artifact(content="t"), "")
    assert v.passed is False
    assert v.score == 0.25
    assert "judge_disagreement" in v.detail


async def test_an_explicit_fail_is_honoured_even_with_a_high_score():
    v = await LLMJudge(_stub({"score": 0.95, "passed": False}), model="m",
                       threshold=0.7).evaluate(TASK, Artifact(content="t"), "")
    assert v.passed is False
    assert "judge_disagreement" in v.detail


async def test_agreement_records_no_disagreement_note():
    v = await LLMJudge(_stub({"score": 0.9, "passed": True}), model="m",
                       threshold=0.7).evaluate(TASK, Artifact(content="t"), "")
    assert v.passed is True and "judge_disagreement" not in v.detail


async def test_vision_judge_has_nothing_left_to_reconcile():
    """2026-08-19: VisionJudge no longer asks for a self-reported `passed` at
    all, so there is no declared claim left to reconcile against its score --
    passed is just score >= threshold, computed from the rubric alone."""
    art = Artifact(kind="image", meta={"image_b64": "QUJD"})
    bad = {
        "subject_present": False,
        "elements": [], "artifacts_present": True, "composition_intact": True,
    }
    v = await VisionJudge(_stub(bad), model="v", threshold=0.7).evaluate(TASK, art, "")
    assert v.passed is False
    assert "judge_disagreement" not in v.detail
