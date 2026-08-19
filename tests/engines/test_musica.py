"""Step 8 -- composite judging, deterministic checks, and the human gate.

MUSICA is the engine that must not overclaim. Most of these tests are about
what it refuses to assert.
"""

import json
import math
import struct
import wave

import pytest

from engines import musica
from opera.bible import ProjectStore
from opera.config import LLMConfig, LoopConfig, OperaConfig, RoleConfig
from opera.errors import JudgeError
from opera.llm.stub import StubLLMClient
from opera.loop import execute_task
from opera.planner import LLMPlanner
from opera.runner import Runner
from opera.schemas import Artifact, Brief, RunStatus, Task, TaskStatus

PLAN = "Key: A minor. Tempo: 96bpm. Intro 8 bars, verse 16, chorus 16."
PASS = '{"score": 0.9, "passed": true, "issues": []}'
FAIL = '{"score": 0.2, "passed": false, "issues": ["the chorus has no lift"]}'


def write_wav(path, *, seconds=2.0, rate=44100, amplitude=0.5, freq=440.0,
              lead_silence=0.0, trail_silence=0.0, clip=False):
    frames = int(seconds * rate)
    lead = int(lead_silence * rate)
    trail = int(trail_silence * rate)
    data = bytearray()
    for i in range(frames):
        if i < lead or i >= frames - trail:
            value = 0
        else:
            value = amplitude * math.sin(2 * math.pi * freq * i / rate)
            value = int(max(-1.0, min(1.0, value)) * 32767)
            if clip and lead <= i < lead + rate // 10:
                value = 32767
        data += struct.pack("<h", value)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(data))
    return str(path)


def audio_artifact(path, *, plan=PLAN, target=None):
    return Artifact(kind="audio", content=plan, path=path, producer="mixer",
                    meta={"target": target} if target else {})


# --- measurement is measurement ----------------------------------------------

def test_analyse_reads_duration_and_level(tmp_path):
    stats = musica.analyse(write_wav(tmp_path / "a.wav", seconds=2.0, amplitude=0.5))
    assert stats.duration_s == pytest.approx(2.0, abs=0.01)
    assert stats.sample_rate == 44100 and stats.channels == 1
    assert stats.peak == pytest.approx(0.5, abs=0.01)
    assert stats.rms > 0.3


def test_analyse_detects_clipping(tmp_path):
    stats = musica.analyse(write_wav(tmp_path / "a.wav", clip=True))
    assert stats.clipped_samples > 0


def test_analyse_detects_leading_and_trailing_silence(tmp_path):
    stats = musica.analyse(write_wav(tmp_path / "a.wav", seconds=3.0,
                                     lead_silence=1.0, trail_silence=0.5))
    assert stats.leading_silence_s == pytest.approx(1.0, abs=0.05)
    assert stats.trailing_silence_s == pytest.approx(0.5, abs=0.05)


def test_analyse_refuses_a_missing_or_broken_file(tmp_path):
    with pytest.raises(musica.AudioUnreadable):
        musica.analyse(tmp_path / "nope.wav")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav at all")
    with pytest.raises(musica.AudioUnreadable):
        musica.analyse(bad)


# --- deterministic checks ----------------------------------------------------

async def test_checks_pass_a_clean_render(tmp_path):
    from opera.judges import DeterministicJudge

    path = write_wav(tmp_path / "a.wav", seconds=10.0)
    judge = DeterministicJudge(musica.build_checks(musica.MusicSpec(duration_s=10.0)))
    v = await judge.evaluate(Task(goal="g", role="mixer", kind="audio"),
                             audio_artifact(path), "")
    assert v.passed is True and v.score == 1.0


async def test_duration_mismatch_is_reported_with_the_numbers(tmp_path):
    from opera.judges import DeterministicJudge

    path = write_wav(tmp_path / "a.wav", seconds=4.0)
    judge = DeterministicJudge(musica.build_checks(
        musica.MusicSpec(duration_s=30.0, duration_tolerance_s=2.0)))
    v = await judge.evaluate(Task(goal="g", role="mixer", kind="audio"),
                             audio_artifact(path), "")
    assert v.passed is False
    assert any("4.00s vs target 30.00s" in i for i in v.issues)


async def test_per_artifact_target_overrides_the_engine_spec(tmp_path):
    from opera.judges import DeterministicJudge

    path = write_wav(tmp_path / "a.wav", seconds=4.0)
    judge = DeterministicJudge(musica.build_checks(musica.MusicSpec(duration_s=30.0)))
    v = await judge.evaluate(Task(goal="g", role="mixer", kind="audio"),
                             audio_artifact(path, target={"duration_s": 4.0}), "")
    assert v.passed is True


async def test_clipping_and_silence_are_caught(tmp_path):
    from opera.judges import DeterministicJudge

    path = write_wav(tmp_path / "a.wav", seconds=6.0, lead_silence=3.0, clip=True)
    judge = DeterministicJudge(musica.build_checks(musica.MusicSpec()))
    v = await judge.evaluate(Task(goal="g", role="mixer", kind="audio"),
                             audio_artifact(path), "")
    labels = " ".join(v.issues)
    assert "clipping" in labels and "leading_silence" in labels


async def test_a_silent_render_fails(tmp_path):
    from opera.judges import DeterministicJudge

    path = write_wav(tmp_path / "a.wav", seconds=2.0, amplitude=0.0)
    judge = DeterministicJudge(musica.build_checks(musica.MusicSpec()))
    v = await judge.evaluate(Task(goal="g", role="mixer", kind="audio"),
                             audio_artifact(path), "")
    assert v.passed is False
    assert any("not_silent" in i for i in v.issues)


async def test_key_and_tempo_are_only_checked_when_a_detector_is_supplied(tmp_path):
    """No detector, no key/tempo claim -- and the coverage label says so."""
    without = musica.build_checks(musica.MusicSpec(key="A minor", bpm=96))
    assert [name for name, _ in without] == [
        "duration", "clipping", "leading_silence", "trailing_silence", "not_silent"]
    assert "key and tempo" not in musica.coverage_label(None)

    with_detector = musica.build_checks(musica.MusicSpec(key="A minor", bpm=96),
                                        detector=lambda p: ("A minor", 96.0))
    assert "key_and_tempo" in [name for name, _ in with_detector]
    assert "key and tempo" in musica.coverage_label(lambda p: (None, None))


async def test_detected_key_mismatch_fails(tmp_path):
    from opera.judges import DeterministicJudge

    path = write_wav(tmp_path / "a.wav", seconds=2.0)
    checks = musica.build_checks(musica.MusicSpec(key="A minor", bpm=96),
                                 detector=lambda p: ("D minor", 120.0))
    v = await DeterministicJudge(checks).evaluate(
        Task(goal="g", role="mixer", kind="audio"), audio_artifact(path), "")
    assert v.passed is False
    assert any("key D minor vs spec A minor" in i for i in v.issues)
    assert any("tempo 120.0 vs spec 96.0" in i for i in v.issues)


# --- the composite is honest about coverage ----------------------------------

async def test_audio_verdict_reports_artifact_plus_plan(tmp_path):
    """The headline claim of this engine: it measured the file and read the
    plan. It did not listen to the music, and does not say it did."""
    path = write_wav(tmp_path / "a.wav", seconds=10.0)
    client = StubLLMClient(default=PASS)
    judge = musica.build_judge(client, model="m", spec=musica.MusicSpec(duration_s=10.0))

    v = await judge.evaluate(Task(goal="g", role="mixer", kind="audio"),
                             audio_artifact(path), "")

    assert v.judged == "artifact+plan"
    assert set(v.detail["members"]) == {"musica-checks", "musica-plan-review"}
    assert v.detail["members"]["musica-plan-review"]["judged"] == "plan"


async def test_plan_only_artifact_reports_plan(tmp_path):
    client = StubLLMClient(default=PASS)
    judge = musica.build_judge(client, model="m")
    v = await judge.evaluate(Task(goal="g", role="composer", kind="text"),
                             Artifact(kind="text", content=PLAN), "")
    assert v.judged == "plan"


async def test_a_failing_check_sinks_the_composite_even_with_a_glowing_review(tmp_path):
    """A deterministic fact is not averaged away by an opinion."""
    path = write_wav(tmp_path / "a.wav", seconds=2.0)
    client = StubLLMClient(default='{"score": 1.0, "passed": true, "issues": []}')
    judge = musica.build_judge(client, model="m",
                               spec=musica.MusicSpec(duration_s=60.0))
    v = await judge.evaluate(Task(goal="g", role="mixer", kind="audio"),
                             audio_artifact(path), "")
    assert v.passed is False
    assert any("[musica-checks]" in i for i in v.issues)


async def test_audio_without_a_plan_fails_loudly_rather_than_scoring_nothing(tmp_path):
    path = write_wav(tmp_path / "a.wav")
    judge = musica.build_judge(StubLLMClient(default=PASS), model="m")
    with pytest.raises(JudgeError, match="no arrangement plan"):
        await judge.evaluate(Task(goal="g", role="mixer", kind="audio"),
                             Artifact(kind="audio", path=path, content="", producer="mixer"), "")


# --- the human gate ----------------------------------------------------------

async def test_gate_holds_a_passing_render_for_a_person(tmp_path):
    from tests.doubles import ScriptedJudge, ScriptedProducer

    task = Task(goal="render it", role="mixer", kind="audio")
    gate = musica.HoldForHuman()
    await execute_task(task, ScriptedProducer(["audio"], name="mixer", kind="audio"),
                       ScriptedJudge([True]), "ctx", gate=gate)
    assert task.status is TaskStatus.AWAITING_REVIEW
    assert gate.held == [task.id]


async def test_callback_gate_lets_a_person_approve(tmp_path):
    from tests.doubles import ScriptedJudge, ScriptedProducer

    seen = []
    gate = musica.CallbackGate(lambda t, a: seen.append(a.id) or True)
    task = Task(goal="render it", role="mixer", kind="audio")
    await execute_task(task, ScriptedProducer(["audio"], name="mixer", kind="audio"),
                       ScriptedJudge([True]), "ctx", gate=gate)
    assert task.status is TaskStatus.DONE and len(seen) == 1


async def test_gate_is_not_reached_when_the_checks_already_failed(tmp_path):
    from tests.doubles import ScriptedJudge, ScriptedProducer

    gate = musica.CallbackGate(lambda t, a: True)
    task = Task(goal="render it", role="mixer", kind="audio")
    await execute_task(task, ScriptedProducer(["audio"], name="mixer", kind="audio"),
                       ScriptedJudge([False]), "ctx", config=LoopConfig(max_attempts=1),
                       gate=gate)
    assert task.status is TaskStatus.FAILED and gate.calls == 0


async def test_a_run_awaiting_review_does_not_report_done(tmp_path):
    plan = json.dumps({"tasks": [{"goal": "Render the cue", "role": "mixer", "kind": "audio"}]})

    class FakeMixer:
        name, kind, available = "mixer", "audio", True

        async def produce(self, brief):
            return Artifact(kind="audio", content=PLAN, path="/tmp/x.wav", producer="mixer")

    client = StubLLMClient([("You are a music producer", plan)], default=PASS)
    spec = musica.build(client, audio_producer=FakeMixer(), require_human=True)
    # Bypass the deterministic file checks; this test is about the gate.
    spec.judge.audio_judge = spec.judge.plan_judge

    store = ProjectStore(tmp_path)
    runner = Runner(spec, store, planner=LLMPlanner(client, spec, model="m"))
    report = await runner.run(store.create("Cue", "musica"), "score the chase")

    assert report.run.tasks[0].status is TaskStatus.AWAITING_REVIEW
    assert report.status is RunStatus.AWAITING_REVIEW


# --- producers ---------------------------------------------------------------

def test_fluidsynth_is_unavailable_without_a_soundfont(tmp_path):
    p = musica.FluidSynthProducer("mixer", RoleConfig(model="m", kind="audio"),
                                  soundfont=tmp_path / "missing.sf2", output_dir=tmp_path)
    assert p.available is False


def test_acestep_is_unavailable_without_its_binary(tmp_path):
    p = musica.AceStepProducer("mixer", RoleConfig(model="m", kind="audio"),
                               output_dir=tmp_path, binary="definitely-not-installed")
    assert p.available is False


async def test_fluidsynth_without_midi_defers(tmp_path):
    from opera.errors import ProducerUnavailable

    sf = tmp_path / "font.sf2"
    sf.write_bytes(b"fake")
    p = musica.FluidSynthProducer("mixer", RoleConfig(model="m", kind="audio"),
                                  soundfont=sf, output_dir=tmp_path, available=True)
    with pytest.raises(ProducerUnavailable, match="MIDI"):
        await p.produce(Brief(task_id="t1", goal="g", kind="audio", role="mixer"))


async def test_renderer_stamps_the_plan_it_rendered_from(tmp_path):
    """The contract the judge depends on."""
    real = musica.AceStepProducer("mixer", RoleConfig(model="m", kind="audio"),
                                  output_dir=tmp_path, available=True)
    brief = Brief(task_id="t1", goal="render it", kind="audio", role="mixer",
                  params={"plan": PLAN})
    assert real._plan_text(brief) == PLAN


async def test_composer_writes_a_plan(tmp_path):
    client = StubLLMClient(default=PLAN)
    cfg = musica.default_roles()["composer"]
    art = await musica.LLMPlanProducer(client, "composer", cfg).produce(
        Brief(task_id="t1", goal="write the theme", kind="text", role="composer"))
    assert art.kind == "text" and art.content == PLAN


# --- vocabulary and routing --------------------------------------------------

def test_musica_vocabulary_is_its_own():
    spec = musica.build(StubLLMClient())
    assert set(spec.roles) == {"composer", "arranger", "mixer"}
    assert spec.kinds == {"text", "audio"}


def test_musica_routing_ignores_incidental_music_talk():
    router = musica.build(StubLLMClient()).router()
    assert router.route("a musical score plays in the background of the diner").role != "mixer"
    assert router.route("render the audio for act two").role == "mixer"
    assert router.route("compose the main theme").role == "composer"


def test_musica_is_registered():
    from opera import registry

    assert "musica" in registry.available()
