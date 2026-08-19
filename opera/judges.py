"""The judge implementations spec 3.2 asks OPERA to ship.

These are engine-agnostic on purpose -- an engine's ``judge.py`` wires them
together with its own prompts and thresholds rather than reimplementing them.

Every verdict states what was actually assessed. A judge that only ever saw a
plan reports ``judged="plan"``; one that saw four sampled frames reports
``judged="frames"``. No judge is permitted to imply coverage it does not have.
"""

from __future__ import annotations

import asyncio
import base64
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence

from .config import LLMConfig
from .errors import JSONParseError, JudgeError
from .llm.parsing import extract_json_object
from .protocols import Judge, LLMClient
from .schemas import Artifact, Task, Verdict

JUDGE_SYSTEM = (
    "You are a strict reviewer. Assess the work against the goal and the project "
    "context. Reply with JSON only:\n"
    '{"score": 0.0-1.0, "passed": true|false, "issues": ["..."]}\n'
    "Issues must be specific and actionable. An empty issues list means you found "
    "nothing to fix. Do not restate the work."
)

VISION_RUBRIC_SYSTEM = (
    "You are a strict image reviewer. You will be shown an image and a goal "
    "describing what it should depict. Do not self-assess a pass/fail or a "
    "numeric score -- report only what you actually observe, as JSON:\n"
    '{"subject_present": true|false, "subject_note": "...", '
    '"elements": [{"element": "<a distinct visual element named in the goal>", '
    '"matched": true|false, "note": "..."}], '
    '"artifacts_present": true|false, "artifacts_note": "...", '
    '"composition_intact": true|false, "composition_note": "..."}\n'
    "List each distinct visual element the goal names (subjects, objects, "
    "setting, style, color) as its own entry in 'elements' with whether it is "
    "actually present in the image. 'artifacts_present' means visible "
    "rendering defects (garbling, warped anatomy, broken geometry, text soup) "
    "-- not stylistic choices you personally dislike. 'composition_intact' "
    "means the image is a coherent picture at all, not corrupted or "
    "unreadable. Be specific in the notes; they become the issue list."
)

Check = Callable[[Artifact], tuple[bool, str]]
AsyncCheck = Callable[[Artifact], Awaitable[tuple[bool, str]]]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _coerce_score(data: dict[str, Any], *, judge_name: str, stage: str) -> float:
    """Parse a model-reported score at the one place every model-backed judge
    reads one, or fail loudly.

    A judge that could not determine a score has not judged anything. Spec 11
    forbids implying coverage a judge doesn't have, and a silent default of 0.0
    is exactly that -- it is recorded as a real (if poor) assessment, and can
    coincidentally clear a low or zero threshold. Missing, null, non-numeric,
    and non-finite (NaN/Infinity -- valid Python `json.loads` output, not just
    a hypothetical) scores are all parse failures, not zeros.

    Out-of-range numbers (1.5, -0.2) are the one case that IS coerced, by
    clamping into [0, 1] -- the model did report a real number, just an invalid
    one, and clamping preserves more signal than discarding it.
    """
    if "score" not in data:
        raise JudgeError(f"judge {judge_name!r} ({stage}) did not report a score")
    raw = data["score"]
    # bool is a subclass of int in Python -- {"score": true} must not silently
    # become 1.0.
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise JudgeError(f"judge {judge_name!r} ({stage}) returned a non-numeric score: {raw!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise JudgeError(f"judge {judge_name!r} ({stage}) returned a non-numeric score: {raw!r}")
    if not math.isfinite(value):
        raise JudgeError(f"judge {judge_name!r} ({stage}) returned a non-finite score: {raw!r}")
    return _clamp(value)


def _reconcile(declared: Any, score: float, threshold: float) -> tuple[bool, dict]:
    """Reconcile a model's `passed` claim with its own score.

    Small local models routinely return `{"score": 0.25, "passed": true}`. Taking
    the flag at face value stores a passing verdict on work the same judge just
    scored well under threshold -- an internally contradictory verdict, and one
    the engine cannot justify (spec 11).

    So a pass must be backed by the score. An explicit `false` is always
    honoured, and any disagreement is recorded rather than quietly resolved.
    """
    meets = score >= threshold
    if not isinstance(declared, bool):
        return meets, {}
    passed = declared and meets
    if declared != meets:
        return passed, {"judge_disagreement":
                        f"model said passed={declared} but scored {score:.2f} "
                        f"against threshold {threshold:.2f}; resolved to {passed}"}
    return passed, {}


def _score_rubric(data: dict[str, Any], *, judge_name: str) -> tuple[float, list[str]]:
    """Compute a VisionJudge score from structured observations, never from a
    self-reported number.

    Confirmed live 2026-08-18: llava:7b returned the identical 0.500 score
    for a goal the image plausibly matched and one it had nothing to do with
    (a fox-in-snow render vs. "a golden retriever on a beach") -- its number
    carried no signal, even though its free text and `passed` boolean
    genuinely did discriminate between the two. Same principle as
    ``Verdict.judged``: derive the verdict from what the model actually
    observed, never accept its self-assessment of the outcome.

    Weights: subject presence and prompt-element coverage are the two things
    that make an image *right*; freedom from artifacts and an intact
    composition are what make it *usable*. Composition does NOT hard-gate the
    score to zero -- confirmed live 2026-08-18 against both llava:7b and
    qwen2.5vl:7b that weak vision models routinely misread
    'composition_intact' as "does this look like the kind of image I
    expected" rather than "is this technically corrupted", and set it False
    on a perfectly coherent image that just isn't a photograph. Gating on a
    field models this unreliable would zero out otherwise-correct verdicts;
    weighting it lets the (empirically more reliable) subject/element signal
    still dominate.
    """
    def _bool(key: str) -> bool:
        if key not in data:
            raise JudgeError(f"judge {judge_name!r} (vision-judge) rubric missing {key!r}")
        val = data[key]
        if not isinstance(val, bool):
            raise JudgeError(
                f"judge {judge_name!r} (vision-judge) rubric {key!r} is not a boolean: {val!r}"
            )
        return val

    def _note(key: str) -> str:
        val = data.get(key)
        return str(val).strip() if val else ""

    composition_intact = _bool("composition_intact")
    subject_present = _bool("subject_present")
    artifacts_present = _bool("artifacts_present")

    elements = data.get("elements")
    if not isinstance(elements, list):
        raise JudgeError(
            f"judge {judge_name!r} (vision-judge) rubric 'elements' is not a list: {elements!r}"
        )
    parsed_elements: list[tuple[str, bool, str]] = []
    for item in elements:
        if (not isinstance(item, dict) or "element" not in item
                or not isinstance(item.get("matched"), bool)):
            raise JudgeError(
                f"judge {judge_name!r} (vision-judge) malformed rubric element: {item!r}"
            )
        parsed_elements.append(
            (str(item["element"]), item["matched"], str(item.get("note", "")).strip())
        )

    issues: list[str] = []
    if not composition_intact:
        note = _note("composition_note")
        issues.append("composition is not intact" + (f": {note}" if note else ""))
    if not subject_present:
        note = _note("subject_note")
        issues.append("subject not present" + (f": {note}" if note else ""))
    for name, matched, note in parsed_elements:
        if not matched:
            issues.append(f"prompt element not matched: {name}" + (f" ({note})" if note else ""))
    if artifacts_present:
        note = _note("artifacts_note")
        issues.append("obvious rendering artifacts present" + (f": {note}" if note else ""))

    score = 0.3 if subject_present else 0.0
    if parsed_elements:
        matched_count = sum(1 for _, matched, _ in parsed_elements if matched)
        score += 0.4 * (matched_count / len(parsed_elements))
    else:
        # Nothing named to check against -- no evidence to penalise, so this
        # component defaults to full credit rather than zero.
        score += 0.4
    score += 0.0 if artifacts_present else 0.15
    score += 0.15 if composition_intact else 0.0

    return _clamp(score), issues


class LLMJudge:
    """Scores a text or code artifact against the goal plus bible context."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        name: str = "llm-judge",
        threshold: float = 0.7,
        timeout_s: float = 120.0,
        system: str = JUDGE_SYSTEM,
        judged: str = "artifact",
        no_think: bool = True,
        max_chars: int = 12000,
    ) -> None:
        if client is None:
            raise JudgeError("LLMJudge requires an explicit LLM client")
        self.client = client
        self.model = model
        self.name = name
        self.threshold = threshold
        self.timeout_s = timeout_s
        self.system = system
        self.judged = judged
        self.no_think = no_think
        self.max_chars = max_chars

    def _prompt(self, task: Task, artifact: Artifact, context: str) -> str:
        parts = [f"Goal: {task.goal}"]
        if context.strip():
            parts.append(f"\nProject context:\n{context.strip()}")
        body = artifact.content[: self.max_chars]
        parts.append(f"\nWork to assess ({artifact.kind}):\n{body}")
        return "\n".join(parts)

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict:
        raw = await self.client.complete(
            system=self.system,
            prompt=self._prompt(task, artifact, context),
            model=self.model,
            format_json=True,
            timeout=self.timeout_s,
            no_think=self.no_think,
        )
        try:
            data = extract_json_object(raw, stage="judge")
        except JSONParseError as exc:
            raise JudgeError(f"judge {self.name!r} returned unparseable output: {exc}") from exc

        score = _coerce_score(data, judge_name=self.name, stage="judge")
        issues = [str(i) for i in (data.get("issues") or []) if str(i).strip()]
        passed, note = _reconcile(data.get("passed"), score, self.threshold)
        return Verdict(
            score=score, passed=passed, issues=issues,
            judged=self.judged, judge_name=self.name,
            detail={"model": self.model, "threshold": self.threshold, **note},
        )


class VisionJudge:
    """Scores an image artifact with a multimodal model.

    The image is passed as base64 on the message, from ``artifact.path`` when
    present or ``artifact.meta['image_b64']`` otherwise.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        name: str = "vision-judge",
        threshold: float = 0.7,
        timeout_s: float = 180.0,
        system: str = VISION_RUBRIC_SYSTEM,
        judged: str = "artifact",
    ) -> None:
        if client is None:
            raise JudgeError("VisionJudge requires an explicit LLM client")
        self.client = client
        self.model = model
        self.name = name
        self.threshold = threshold
        self.timeout_s = timeout_s
        self.system = system
        self.judged = judged

    @staticmethod
    def _image_b64(artifact: Artifact) -> str:
        if b64 := artifact.meta.get("image_b64"):
            return str(b64)
        if artifact.path:
            path = Path(artifact.path)
            if path.exists():
                return base64.b64encode(path.read_bytes()).decode("ascii")
        raise JudgeError(f"artifact {artifact.id} has no readable image to judge")

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict:
        prompt_used = artifact.meta.get("prompt", "")
        parts = [f"Goal: {task.goal}"]
        if prompt_used:
            parts.append(f"Generation prompt: {prompt_used}")
        if context.strip():
            parts.append(f"\nProject context:\n{context.strip()}")
        parts.append("\nAssess the attached image for prompt adherence, composition, "
                     "and consistency with the project context.")

        raw = await self.client.complete(
            system=self.system,
            prompt="\n".join(parts),
            model=self.model,
            images=[self._image_b64(artifact)],
            format_json=True,
            timeout=self.timeout_s,
        )
        try:
            data = extract_json_object(raw, stage="vision-judge")
        except JSONParseError as exc:
            raise JudgeError(f"judge {self.name!r} returned unparseable output: {exc}") from exc
        score, issues = _score_rubric(data, judge_name=self.name)
        passed = score >= self.threshold
        return Verdict(
            score=score,
            passed=passed,
            issues=issues,
            judged=self.judged,
            judge_name=self.name,
            detail={"model": self.model, "threshold": self.threshold, "rubric": data},
        )


@dataclass
class FrameSample:
    index: int
    b64: str


class FrameSampleJudge:
    """Samples N frames from a video and wraps a VisionJudge over them.

    Scores prompt adherence and cross-frame continuity **only** -- it has not
    watched the video, so it reports ``judged="frames"`` and says so in the
    verdict detail. Motion, pacing and audio are explicitly out of its coverage.
    """

    name = "frame-sample-judge"

    def __init__(
        self,
        vision_judge: VisionJudge,
        *,
        frames: int = 4,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        threshold: float = 0.7,
        aggregate: str = "mean",
        name: str | None = None,
    ) -> None:
        self.vision = vision_judge
        self.frames = max(1, frames)
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.threshold = threshold
        self.aggregate = aggregate
        if name:
            self.name = name

    def available(self) -> bool:
        return shutil.which(self.ffmpeg) is not None

    def _duration(self, path: str) -> float | None:
        if shutil.which(self.ffprobe) is None:
            return None
        proc = subprocess.run(
            [self.ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True,
        )
        try:
            value = float(proc.stdout.strip())
        except ValueError:
            return None
        return value if value > 0 else None

    def _extract(self, path: str) -> list[FrameSample]:
        if shutil.which(self.ffmpeg) is None:
            raise JudgeError(f"{self.ffmpeg} not found; cannot sample frames")

        duration = self._duration(path)
        with tempfile.TemporaryDirectory() as td:
            if duration is not None:
                # Evenly spaced seeks. Continuity is the thing being assessed,
                # so the frames must span the clip rather than cluster.
                for i in range(self.frames):
                    ts = duration * (i + 0.5) / self.frames
                    proc = subprocess.run(
                        [self.ffmpeg, "-v", "error", "-ss", f"{ts:.3f}", "-i", path,
                         "-frames:v", "1", "-vf", "scale=768:-2", "-y",
                         str(Path(td) / f"frame_{i:03d}.png")],
                        capture_output=True, text=True,
                    )
                    if proc.returncode != 0:
                        raise JudgeError(
                            f"ffmpeg failed sampling {path} at {ts:.3f}s: "
                            f"{proc.stderr.strip()[:300]}"
                        )
            else:
                proc = subprocess.run(
                    [self.ffmpeg, "-v", "error", "-i", path, "-vf", "thumbnail,scale=768:-2",
                     "-frames:v", str(self.frames), "-y", str(Path(td) / "frame_%03d.png")],
                    capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    raise JudgeError(f"ffmpeg failed sampling {path}: {proc.stderr.strip()[:300]}")

            samples = [
                FrameSample(index=i, b64=base64.b64encode(f.read_bytes()).decode())
                for i, f in enumerate(sorted(Path(td).glob("frame_*.png")))
            ]
            if not samples:
                raise JudgeError(f"ffmpeg produced no frames from {path}")
            return samples

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict:
        if not artifact.path:
            raise JudgeError(f"artifact {artifact.id} has no video path to sample")
        samples = await asyncio.to_thread(self._extract, artifact.path)

        verdicts: list[Verdict] = []
        for sample in samples:
            frame = Artifact(
                task_id=artifact.task_id, kind="image", producer=artifact.producer,
                meta={"image_b64": sample.b64, "prompt": artifact.meta.get("prompt", ""),
                      "frame_index": sample.index},
            )
            verdicts.append(await self.vision.evaluate(task, frame, context))

        scores = [v.score for v in verdicts]
        score = min(scores) if self.aggregate == "min" else sum(scores) / len(scores)
        issues: list[str] = []
        for idx, v in enumerate(verdicts):
            for issue in v.issues:
                label = f"frame {idx}: {issue}"
                if label not in issues:
                    issues.append(label)
        return Verdict(
            score=_clamp(score),
            passed=score >= self.threshold and all(v.passed for v in verdicts),
            issues=issues,
            judged="frames",
            judge_name=self.name,
            detail={
                "frames_sampled": len(samples),
                "per_frame_scores": scores,
                "aggregate": self.aggregate,
                "coverage": "prompt adherence and cross-frame continuity only; "
                            "motion, pacing and audio were not assessed",
            },
        )


class DeterministicJudge:
    """Runs objective checks. No model, no opinion.

    Each check returns ``(ok, message)``. The score is the pass fraction, and
    ``passed`` requires every check to pass -- a deterministic check that fails
    is a fact, not a preference to be averaged away.
    """

    def __init__(
        self,
        checks: Sequence[tuple[str, Check]] | Sequence[Check],
        *,
        name: str = "deterministic-judge",
        judged: str = "artifact",
    ) -> None:
        self.checks: list[tuple[str, Check]] = []
        for item in checks:
            if isinstance(item, tuple):
                self.checks.append(item)
            else:
                self.checks.append((getattr(item, "__name__", "check"), item))
        self.name = name
        self.judged = judged

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict:
        if not self.checks:
            return Verdict(score=1.0, passed=True, issues=[], judged=self.judged,
                           judge_name=self.name, detail={"checks": 0})
        results: list[tuple[str, bool, str]] = []
        for label, check in self.checks:
            try:
                ok, message = check(artifact)
            except Exception as exc:  # a broken check is a failed check, loudly
                ok, message = False, f"check raised {type(exc).__name__}: {exc}"
            results.append((label, bool(ok), str(message)))

        failed = [(label, msg) for label, ok, msg in results if not ok]
        score = 1.0 - (len(failed) / len(results))
        return Verdict(
            score=_clamp(score),
            passed=not failed,
            issues=[f"{label}: {msg}" for label, msg in failed],
            judged=self.judged,
            judge_name=self.name,
            detail={"checks": len(results),
                    "results": {label: ok for label, ok, _ in results}},
        )


class CompositeJudge:
    """Runs several judges and combines them under a stated policy.

    ``judged`` is the honest union of what the members actually looked at, so a
    composite of a deterministic file check and an LLM plan review reports
    ``"checks+plan"`` -- never ``"artifact"``.
    """

    POLICIES = ("all_must_pass", "any_may_pass", "weighted")

    def __init__(
        self,
        judges: Iterable[Judge],
        *,
        name: str = "composite-judge",
        policy: str = "all_must_pass",
        weights: Sequence[float] | None = None,
        threshold: float = 0.7,
    ) -> None:
        self.judges = list(judges)
        if not self.judges:
            raise JudgeError("CompositeJudge needs at least one judge")
        if policy not in self.POLICIES:
            raise JudgeError(f"unknown policy {policy!r}; expected one of {self.POLICIES}")
        self.name = name
        self.policy = policy
        self.threshold = threshold
        if weights is not None and len(weights) != len(self.judges):
            raise JudgeError("weights must match the number of judges")
        self.weights = list(weights) if weights else [1.0] * len(self.judges)

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict:
        verdicts = [await j.evaluate(task, artifact, context) for j in self.judges]

        # A subordinate's own `.passed` is just as much a claim as a raw
        # model's `passed` field was in the original bug -- a hand-rolled or
        # third-party Judge is not guaranteed to have reconciled it against
        # its own score before returning. Route every subordinate through the
        # same reconciliation path CompositeJudge's own callers get, judged
        # against that subordinate's own configured threshold when it has one
        # (falling back to the composite's), so a looser subordinate isn't
        # silently held to a stricter bar it never agreed to.
        reconciled: list[bool] = []
        notes: dict[str, dict] = {}
        for j, v in zip(self.judges, verdicts):
            sub_threshold = getattr(j, "threshold", self.threshold)
            effective_passed, note = _reconcile(v.passed, v.score, sub_threshold)
            reconciled.append(effective_passed)
            notes[v.judge_name] = note

        total_weight = sum(self.weights) or 1.0
        weighted = sum(v.score * w for v, w in zip(verdicts, self.weights)) / total_weight

        if self.policy == "all_must_pass":
            passed = all(reconciled)
            score = min(v.score for v in verdicts)
        elif self.policy == "any_may_pass":
            passed = any(reconciled)
            score = max(v.score for v in verdicts)
        else:
            score = weighted
            passed = score >= self.threshold

        issues: list[str] = []
        for v in verdicts:
            for issue in v.issues:
                label = f"[{v.judge_name}] {issue}"
                if label not in issues:
                    issues.append(label)

        # Do not let the composite imply coverage none of its members had.
        seen: list[str] = []
        for v in verdicts:
            if v.judged not in seen:
                seen.append(v.judged)

        return Verdict(
            score=_clamp(score),
            passed=passed,
            issues=issues,
            judged="+".join(seen),
            judge_name=self.name,
            detail={
                "policy": self.policy,
                "members": {
                    v.judge_name: {
                        "score": v.score,
                        "declared_passed": v.passed,
                        "passed": eff,
                        "judged": v.judged,
                        **notes[v.judge_name],
                    }
                    for v, eff in zip(verdicts, reconciled)
                },
                "weighted_score": weighted,
            },
        )
