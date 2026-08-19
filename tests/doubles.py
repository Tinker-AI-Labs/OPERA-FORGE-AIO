"""Scripted producers, judges and planners.

Everything in the suite runs without Ollama, ComfyUI or ffmpeg (spec 9).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Iterable

from opera.errors import ProducerUnavailable
from opera.planner import Plan, PlannedTask
from opera.schemas import Artifact, Brief, Task, Verdict


class ScriptedProducer:
    """Returns canned content, one item per production."""

    def __init__(
        self,
        contents: Iterable[str] | None = None,
        *,
        name: str = "scripted",
        kind: str = "text",
        available: bool = True,
        raises: Exception | None = None,
        raise_on_attempt: int = 1,
        delay: float = 0.0,
    ) -> None:
        self.contents = list(contents or ["draft-1", "draft-2", "draft-3", "draft-4"])
        self.name = name
        self.kind = kind
        self.available = available
        self.raises = raises
        self.raise_on_attempt = raise_on_attempt
        self.delay = delay
        self.briefs: list[Brief] = []

    @property
    def calls(self) -> int:
        return len(self.briefs)

    async def produce(self, brief: Brief) -> Artifact:
        self.briefs.append(brief)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises is not None and self.calls == self.raise_on_attempt:
            raise self.raises
        idx = min(self.calls - 1, len(self.contents) - 1)
        return Artifact(kind=self.kind, content=self.contents[idx], producer=self.name)


class UnavailableProducer:
    def __init__(self, name: str = "offline", kind: str = "image") -> None:
        self.name = name
        self.kind = kind
        self.available = False
        self.calls = 0

    async def produce(self, brief: Brief) -> Artifact:  # pragma: no cover
        self.calls += 1
        raise AssertionError("an unavailable producer must never be asked to produce")


class GoesOfflineProducer:
    """Available at check time, gone by production time."""

    def __init__(self, name: str = "flaky", kind: str = "audio") -> None:
        self.name = name
        self.kind = kind
        self.available = True
        self.calls = 0

    async def produce(self, brief: Brief) -> Artifact:
        self.calls += 1
        raise ProducerUnavailable("comfyui went away mid-run")


class ScriptedJudge:
    """Yields verdicts in order; repeats the last one once exhausted."""

    def __init__(
        self,
        verdicts: Iterable[Verdict | bool | float] | None = None,
        *,
        name: str = "scripted-judge",
        judged: str = "artifact",
    ) -> None:
        self.name = name
        self.judged = judged
        self._script = list(verdicts or [True])
        self.seen: list[tuple[Task, Artifact, str]] = []

    @property
    def calls(self) -> int:
        return len(self.seen)

    def _verdict(self, i: int) -> Verdict:
        item = self._script[min(i, len(self._script) - 1)]
        if isinstance(item, Verdict):
            return item
        if isinstance(item, bool):
            return Verdict(
                score=1.0 if item else 0.2,
                passed=item,
                issues=[] if item else ["not good enough"],
                judged=self.judged,
                judge_name=self.name,
            )
        return Verdict(
            score=float(item),
            passed=float(item) >= 0.7,
            issues=[] if float(item) >= 0.7 else ["below threshold"],
            judged=self.judged,
            judge_name=self.name,
        )

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict:
        self.seen.append((task, artifact, context))
        return self._verdict(len(self.seen) - 1)


class ExplodingJudge:
    name = "exploding-judge"

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict:
        raise RuntimeError("vision model crashed")


class StaticPlanner:
    """Emits a fixed task list, bypassing any LLM."""

    def __init__(self, tasks: list[dict[str, Any]], *, notes: list[str] | None = None) -> None:
        self._tasks = tasks
        self._notes = notes or []

    async def plan(self, goal: str, context: str = "") -> Plan:
        return Plan(
            tasks=[PlannedTask(**t) for t in self._tasks],
            notes=list(self._notes),
        )


class CallablePlanner:
    def __init__(self, fn: Callable[[str, str], Plan]) -> None:
        self._fn = fn

    async def plan(self, goal: str, context: str = "") -> Plan:
        return self._fn(goal, context)


class ApprovingGate:
    name = "auto-approve"

    def __init__(self, approve: bool = True) -> None:
        self._approve = approve
        self.calls = 0

    async def approve(self, task: Task, artifact: Artifact) -> bool:
        self.calls += 1
        return self._approve
