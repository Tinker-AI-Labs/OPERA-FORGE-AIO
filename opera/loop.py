"""The produce -> judge -> revise cycle for a single task.

Every rule in spec section 4 that applies to one task lives here, and each is
marked with the rule number it implements. The loop talks to the ``Producer``
and ``Judge`` protocols only -- it never names a concrete reviewer, which is the
seam that would let a deterministic verifier be dropped in later (spec 1).

The loop holds no state on ``self``. Everything it needs arrives as an argument
and everything it produces is returned or written onto the passed ``Task``
(spec 4.7).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import dataclass

from .config import LoopConfig, RoleConfig
from .errors import JudgeError, ProducerError, ProducerUnavailable
from .protocols import HumanGate, Judge, Producer
from .schemas import Artifact, Brief, Task, TaskStatus, Verdict


@dataclass(frozen=True)
class LoopOutcome:
    """What one task's cycle actually did. Telemetry, not narrative."""

    task: Task
    productions: int
    duration_s: float
    passed: bool


def _limits(config: LoopConfig, role: RoleConfig | None) -> tuple[int, float]:
    max_attempts = role.max_attempts if role is not None else config.max_attempts
    threshold = role.pass_threshold if role is not None else config.pass_threshold
    return max(1, int(max_attempts)), float(threshold)


def _policy(config: LoopConfig, role: RoleConfig | None) -> str:
    return role.policy if role is not None else config.policy


def _ceiling(config: LoopConfig, role: RoleConfig | None) -> float | None:
    return role.ceiling if role is not None else config.ceiling


async def execute_task(
    task: Task,
    producer: Producer,
    judge: Judge,
    context: str,
    *,
    config: LoopConfig | None = None,
    role_config: RoleConfig | None = None,
    gate: HumanGate | None = None,
) -> LoopOutcome:
    """Run one task to a final, judged artifact.

    Returns with ``task.artifact`` being the artifact that ``task.verdict``
    describes -- always the same bytes, never a successor (spec 4.1). Which
    policy decides *how* that artifact is chosen -- see ``_execute_threshold``
    (revise until a verdict passes) and ``_execute_best_of_n`` (generate a
    batch, keep the best) -- but the guarantee is the same either way.
    """
    config = config or LoopConfig()
    max_attempts, threshold = _limits(config, role_config)
    started = time.monotonic()

    task.started_at = task.started_at or _utcnow()
    task.status = TaskStatus.RUNNING

    # Spec 3.1 / 11: an unavailable backend defers the task. It never produces
    # placeholder or synthetic output, and it never fabricates a verdict.
    if not getattr(producer, "available", False):
        return _defer(
            task,
            f"producer {getattr(producer, 'name', '?')!r} is unavailable",
            started,
        )

    if _policy(config, role_config) == "best_of_n":
        outcome = await _execute_best_of_n(
            task, producer, judge, context, max_attempts,
            _ceiling(config, role_config), started,
        )
    else:
        outcome = await _execute_threshold(
            task, producer, judge, context, max_attempts, config, started,
        )

    # The gate is an engine decision applied after the cycle, not a step baked
    # into the loop (MUSICA's human gate, spec 2).
    await apply_gate(task, gate)
    return outcome


async def _execute_threshold(
    task: Task, producer: Producer, judge: Judge, context: str,
    max_attempts: int, config: LoopConfig, started: float,
) -> LoopOutcome:
    """Revise until a verdict passes or the attempt budget runs out."""
    verdict: Verdict | None = None
    artifact: Artifact | None = None

    while task.attempts < max_attempts:
        brief = Brief(
            task_id=task.id,
            goal=task.goal,
            kind=task.kind,
            role=task.role,
            context=context,
            attempt=task.attempts + 1,
            prior=artifact,
            issues=list(verdict.issues) if verdict is not None else [],
        )

        try:
            artifact = await producer.produce(brief)
        except ProducerUnavailable as exc:
            # Discovered mid-flight. Anything already produced and judged is
            # kept; the task is honestly marked deferred rather than done.
            return _defer(task, str(exc) or "producer became unavailable", started)

        if artifact is None:
            raise ProducerError(
                f"producer {getattr(producer, 'name', '?')!r} returned no artifact for task {task.id}"
            )

        # Spec 4.2: max_attempts counts total productions, not revisions.
        task.attempts += 1
        artifact.task_id = task.id
        artifact.attempt = task.attempts
        if not artifact.producer:
            artifact.producer = getattr(producer, "name", "")
        if not artifact.kind:
            artifact.kind = task.kind

        try:
            verdict = await judge.evaluate(task, artifact, context)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
            raise JudgeError(
                f"judge {getattr(judge, 'name', '?')!r} failed on task {task.id}: {exc}"
            ) from exc

        # Spec 4.1: the verdict is attached to the artifact it assessed, and the
        # pair is stored together. The loop can only exit from here, so there is
        # no path on which a stored artifact lacks its own verdict.
        artifact.verdict = verdict
        task.artifacts.append(artifact)

        if verdict.passed:
            break
        if config.revise_only_with_issues and not verdict.issues:
            # Nothing actionable to revise against; another pass would just be
            # a re-roll charged against the attempt budget.
            break

    return _finish(task, verdict, started, max_attempts)


async def _execute_best_of_n(
    task: Task, producer: Producer, judge: Judge, context: str,
    max_attempts: int, ceiling: float | None, started: float,
) -> LoopOutcome:
    """Generate up to ``max_attempts`` independent productions, judge each,
    and keep the highest-scoring one as the task's canonical result.

    For a judge that ranks reliably but doesn't calibrate to an absolute pass
    bar (2026-08-19: VisionJudge's rubric scores a genuinely good ARTISTA
    image around 0.567-0.7 against a real vision model, never clearing a
    typical 0.7 threshold), threshold-gating fails everything, including
    good work. A judge that ranks correctly is still useful -- just not as a
    gate. Every attempt is judged and appended to ``task.artifacts`` exactly
    like the threshold path (spec 4.1 still holds); a losing attempt is not
    discarded, it just isn't the one ``task.artifact`` points at (see
    ``Task.chosen_artifact_id``). Reuses the same never-discard shape
    ``Runner.rerun_task`` already established, rather than inventing a
    second one.

    Attempts are independent draws, not revisions -- there is no single
    "prior" to hand the producer and no issues to react to, so ``Brief``
    always carries ``prior=None`` here.
    """
    attempts_this_call: list[Artifact] = []

    while task.attempts < max_attempts:
        brief = Brief(
            task_id=task.id, goal=task.goal, kind=task.kind, role=task.role,
            context=context, attempt=task.attempts + 1, prior=None, issues=[],
        )

        try:
            artifact = await producer.produce(brief)
        except ProducerUnavailable as exc:
            return _defer(task, str(exc) or "producer became unavailable", started)

        if artifact is None:
            raise ProducerError(
                f"producer {getattr(producer, 'name', '?')!r} returned no artifact for task {task.id}"
            )

        task.attempts += 1
        artifact.task_id = task.id
        artifact.attempt = task.attempts
        if not artifact.producer:
            artifact.producer = getattr(producer, "name", "")
        if not artifact.kind:
            artifact.kind = task.kind

        try:
            verdict = await judge.evaluate(task, artifact, context)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
            raise JudgeError(
                f"judge {getattr(judge, 'name', '?')!r} failed on task {task.id}: {exc}"
            ) from exc

        artifact.verdict = verdict
        task.artifacts.append(artifact)
        attempts_this_call.append(artifact)

        # Not "always spend N": a configured ceiling lets an early, clearly
        # good enough attempt stop the batch rather than paying for the rest
        # of the budget regardless.
        if ceiling is not None and verdict.score >= ceiling:
            break

    winner = max(attempts_this_call, key=lambda a: a.verdict.score)
    task.chosen_artifact_id = winner.id
    return _finish(task, winner.verdict, started, max_attempts)


def _finish(
    task: Task,
    verdict: Verdict | None,
    started: float,
    max_attempts: int,
) -> LoopOutcome:
    duration = time.monotonic() - started
    task.finished_at = _utcnow()

    if verdict is None:  # pragma: no cover - unreachable while max_attempts >= 1
        task.status = TaskStatus.FAILED
        task.error = "loop produced no artifact"
        return LoopOutcome(task=task, productions=task.attempts, duration_s=duration, passed=False)

    if verdict.passed:
        task.status = TaskStatus.DONE
    else:
        # Distinguishable from an exception by the prefix: the task ran to
        # completion, the output just did not clear the bar. The artifact is
        # kept so the work is not lost.
        task.status = TaskStatus.FAILED
        issues = "; ".join(verdict.issues[:3]) or "no issues reported"
        task.error = (
            f"review_not_passed: {task.attempts}/{max_attempts} attempts, "
            f"final score {verdict.score:.2f} ({verdict.judged}); {issues}"
        )

    return LoopOutcome(
        task=task,
        productions=task.attempts,
        duration_s=duration,
        passed=bool(verdict.passed),
    )


def _defer(task: Task, reason: str, started: float) -> LoopOutcome:
    task.status = TaskStatus.DEFERRED
    task.deferred_reason = reason
    task.finished_at = _utcnow()
    return LoopOutcome(
        task=task,
        productions=task.attempts,
        duration_s=time.monotonic() - started,
        passed=False,
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def apply_gate(task: Task, gate: HumanGate | None) -> Task:
    """Run an engine's human gate over a task that otherwise passed.

    Kept separate from ``execute_task`` so the gate is an engine decision the
    runner applies, not something baked into the loop.
    """
    if gate is None or task.status is not TaskStatus.DONE:
        return task
    artifact = task.artifact
    if artifact is None:
        return task
    if not await gate.approve(task, artifact):
        task.status = TaskStatus.AWAITING_REVIEW
    return task
