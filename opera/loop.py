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
    describes -- always the same bytes, never a successor (spec 4.1).
    """
    config = config or LoopConfig()
    max_attempts, _threshold = _limits(config, role_config)
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

    outcome = _finish(task, verdict, started, max_attempts)
    # The gate is an engine decision applied after the cycle, not a step baked
    # into the loop (MUSICA's human gate, spec 2).
    await apply_gate(task, gate)
    return outcome


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
