"""MUSICA's human gate.

A deterministic check and a plan review together still cannot tell you whether
a piece of music is any good. The gate is where that judgement is handed to a
person -- and until a person makes it, the task sits at ``awaiting_review``
rather than claiming to be done.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from opera.schemas import Artifact, Task

Approver = Callable[[Task, Artifact], bool | Awaitable[bool]]


class HoldForHuman:
    """The honest default: nothing passes until a person says so."""

    name = "hold-for-human"

    def __init__(self) -> None:
        self.held: list[str] = []

    async def approve(self, task: Task, artifact: Artifact) -> bool:
        self.held.append(task.id)
        return False


class CallbackGate:
    """Delegates to a supplied approver (a CLI prompt, an API call, a test)."""

    def __init__(self, approver: Approver, *, name: str = "callback-gate") -> None:
        self.approver = approver
        self.name = name
        self.calls = 0

    async def approve(self, task: Task, artifact: Artifact) -> bool:
        self.calls += 1
        result = self.approver(task, artifact)
        if hasattr(result, "__await__"):
            result = await result
        return bool(result)


class AutoApproveGate:
    """Explicitly opts out of human review. Only for tests and unattended runs."""

    name = "auto-approve"

    async def approve(self, task: Task, artifact: Artifact) -> bool:
        return True
