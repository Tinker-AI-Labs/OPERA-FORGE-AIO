"""MUSICA's judge: deterministic checks + a plan review, under a human gate.

This is the engine the ``judged`` field exists for. MUSICA cannot assess a
rendered waveform's *musicality* -- nothing local here can. So it measures what
is measurable, reviews the plan with a language model, and reports
``judged="artifact+plan"`` with the coverage spelled out. It never emits a score
that implies it listened (spec 3.2, 11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from opera.errors import JudgeError
from opera.judges import CompositeJudge, DeterministicJudge, LLMJudge
from opera.protocols import Judge, LLMClient
from opera.schemas import Artifact, Task, Verdict

from .analysis import AudioStats, AudioUnreadable, analyse

MUSICA_PLAN_SYSTEM = (
    "You review a musical arrangement plan -- structure, instrumentation, key, "
    "tempo and how the sections develop. You are reviewing the PLAN, not audio; "
    "do not comment on how it sounds.\n"
    "Reply with JSON only:\n"
    '{"score": 0.0-1.0, "passed": true|false, "issues": ["..."]}'
)

# What a KeyTempoDetector must look like if one is supplied. Without it, key and
# tempo simply are not checked, and the verdict says so.
KeyTempoDetector = Callable[[str], tuple[str | None, float | None]]


@dataclass(frozen=True)
class MusicSpec:
    """The objective targets a render is measured against."""

    duration_s: float | None = None
    duration_tolerance_s: float = 2.0
    key: str | None = None
    bpm: float | None = None
    bpm_tolerance: float = 2.0
    max_leading_silence_s: float = 0.5
    max_trailing_silence_s: float = 2.0
    max_clipping_ratio: float = 0.0001
    min_rms: float = 0.01

    def merged_with(self, meta: dict) -> "MusicSpec":
        """Per-artifact targets override the engine defaults."""
        target = meta.get("target") or {}
        if not isinstance(target, dict):
            return self
        fields = {f: getattr(self, f) for f in self.__dataclass_fields__}
        for key, value in target.items():
            if key in fields and value is not None:
                fields[key] = value
        return MusicSpec(**fields)


def _stats(artifact: Artifact) -> AudioStats:
    if not artifact.path:
        raise AudioUnreadable(f"artifact {artifact.id} has no audio path")
    return analyse(artifact.path)


def build_checks(spec: MusicSpec, detector: KeyTempoDetector | None = None):
    """The deterministic check list. Each returns ``(ok, message)``."""

    def duration(artifact: Artifact) -> tuple[bool, str]:
        s = _stats(artifact)
        target = spec.merged_with(artifact.meta).duration_s
        if target is None:
            return True, "no duration target set"
        delta = abs(s.duration_s - target)
        tol = spec.merged_with(artifact.meta).duration_tolerance_s
        return (delta <= tol,
                f"{s.duration_s:.2f}s vs target {target:.2f}s (tolerance {tol:.2f}s)")

    def clipping(artifact: Artifact) -> tuple[bool, str]:
        s = _stats(artifact)
        limit = spec.merged_with(artifact.meta).max_clipping_ratio
        return (s.clipping_ratio <= limit,
                f"{s.clipped_samples} clipped samples ({s.clipping_ratio:.5f} of total, "
                f"peak {s.peak:.3f})")

    def leading_silence(artifact: Artifact) -> tuple[bool, str]:
        s = _stats(artifact)
        limit = spec.merged_with(artifact.meta).max_leading_silence_s
        return s.leading_silence_s <= limit, f"{s.leading_silence_s:.2f}s of leading silence"

    def trailing_silence(artifact: Artifact) -> tuple[bool, str]:
        s = _stats(artifact)
        limit = spec.merged_with(artifact.meta).max_trailing_silence_s
        return s.trailing_silence_s <= limit, f"{s.trailing_silence_s:.2f}s of trailing silence"

    def not_silent(artifact: Artifact) -> tuple[bool, str]:
        s = _stats(artifact)
        limit = spec.merged_with(artifact.meta).min_rms
        return s.rms >= limit, f"rms {s.rms:.4f} (floor {limit})"

    checks: list[tuple[str, Callable[[Artifact], tuple[bool, str]]]] = [
        ("duration", duration),
        ("clipping", clipping),
        ("leading_silence", leading_silence),
        ("trailing_silence", trailing_silence),
        ("not_silent", not_silent),
    ]

    if detector is not None:
        def key_and_tempo(artifact: Artifact) -> tuple[bool, str]:
            merged = spec.merged_with(artifact.meta)
            if merged.key is None and merged.bpm is None:
                return True, "no key or tempo target set"
            if not artifact.path:
                return False, "no audio path to analyse"
            detected_key, detected_bpm = detector(artifact.path)
            problems = []
            if merged.key and detected_key and detected_key.lower() != merged.key.lower():
                problems.append(f"key {detected_key} vs spec {merged.key}")
            if merged.bpm and detected_bpm and abs(detected_bpm - merged.bpm) > merged.bpm_tolerance:
                problems.append(f"tempo {detected_bpm:.1f} vs spec {merged.bpm:.1f}")
            return (not problems, "; ".join(problems) or
                    f"key {detected_key}, tempo {detected_bpm}")

        checks.append(("key_and_tempo", key_and_tempo))

    return checks


def coverage_label(detector: KeyTempoDetector | None) -> str:
    """What the deterministic pass actually covered. Stated, not implied."""
    covered = "duration, level, clipping and silence"
    if detector is not None:
        covered += ", key and tempo"
    return covered


class MusicaJudge:
    """Audio gets checks + a plan review; a plan alone gets only the review."""

    name = "musica-judge"

    def __init__(self, audio_judge: Judge, plan_judge: Judge, *,
                 audio_kinds: frozenset[str] = frozenset({"audio"})) -> None:
        self.audio_judge = audio_judge
        self.plan_judge = plan_judge
        self.audio_kinds = audio_kinds

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict:
        if artifact.kind not in self.audio_kinds:
            return await self.plan_judge.evaluate(task, artifact, context)

        # Contract: an audio artifact carries the arrangement plan it was
        # rendered from in ``content``. Without it the plan review would be
        # scoring an empty string, and the composite would report coverage it
        # does not have -- fail loudly instead.
        if not (artifact.content or "").strip():
            raise JudgeError(
                f"audio artifact {artifact.id} from producer {artifact.producer!r} carries no "
                "arrangement plan in `content`; MUSICA cannot review what it cannot read"
            )
        return await self.audio_judge.evaluate(task, artifact, context)


def build_judge(
    client: LLMClient,
    *,
    model: str,
    spec: MusicSpec | None = None,
    detector: KeyTempoDetector | None = None,
    threshold: float = 0.7,
) -> MusicaJudge:
    spec = spec or MusicSpec()
    checks = DeterministicJudge(
        build_checks(spec, detector),
        name="musica-checks",
        judged="artifact",   # these DID measure the file itself
    )
    plan_review = LLMJudge(
        client, model=model, name="musica-plan-review", threshold=threshold,
        system=MUSICA_PLAN_SYSTEM,
        # The model reads the plan text, never the audio. Saying "artifact" here
        # would claim coverage this judge does not have.
        judged="plan",
    )
    audio = CompositeJudge(
        [checks, plan_review],
        name="musica-composite",
        policy="all_must_pass",
    )
    return MusicaJudge(audio_judge=audio, plan_judge=plan_review)
