"""Data contracts for OPERA.

Deliberately free of engine vocabulary. ``kind`` and ``role`` are plain strings
validated at runtime against the active ``EngineSpec`` -- there is no
``Literal["text","code",...]`` here, because that would make the core know what
an engine is allowed to produce (spec 3.3, 11).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DEFERRED = "deferred"
    AWAITING_REVIEW = "awaiting_review"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    PARTIAL = "partial"
    FAILED = "failed"
    DEFERRED = "deferred"
    AWAITING_REVIEW = "awaiting_review"


class Verdict(BaseModel):
    """A judge's assessment of one artifact.

    ``judged`` is load-bearing, not decorative: it states what was actually
    assessed so a consumer can tell a score that covers the artifact from one
    that only covers its plan (spec 3.2).
    """

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    issues: list[str] = Field(default_factory=list)
    judged: str
    judge_name: str
    detail: dict[str, Any] = Field(default_factory=dict)
    judged_at: datetime = Field(default_factory=_now)


class Artifact(BaseModel):
    """One production. Carries the verdict on *these* bytes (spec 4.1)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _uid("art"))
    task_id: str = ""
    kind: str = "text"
    producer: str = ""
    attempt: int = 1
    content: str = ""
    path: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    verdict: Verdict | None = None
    created_at: datetime = Field(default_factory=_now)

    def excerpt(self, limit: int = 600) -> str:
        body = (self.content or "").strip()
        if not body and self.path:
            return f"[{self.kind} artifact at {self.path}]"
        if len(body) <= limit:
            return body
        return body[:limit].rstrip() + " ..."


class Brief(BaseModel):
    """The narrow surface handed to a Producer.

    Producers never see the whole Project (spec 3.1) -- only the goal, the
    rendered context string, and, on a revision pass, what they made last time
    plus what the judge objected to.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    goal: str
    kind: str
    role: str
    context: str = ""
    attempt: int = 1
    prior: Artifact | None = None
    issues: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_revision(self) -> bool:
        return self.prior is not None


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _uid("t"))
    goal: str
    role: str
    kind: str
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    artifacts: list[Artifact] = Field(default_factory=list)
    error: str | None = None
    deferred_reason: str | None = None
    corrections: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def artifact(self) -> Artifact | None:
        """The final artifact -- the one whose verdict the task is reported on."""
        return self.artifacts[-1] if self.artifacts else None

    @property
    def verdict(self) -> Verdict | None:
        art = self.artifact
        return art.verdict if art else None


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _uid("run"))
    project_id: str = ""
    engine: str = ""
    goal: str = ""
    tasks: list[Task] = Field(default_factory=list)
    status: RunStatus = RunStatus.PENDING
    error: str | None = None
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.tasks:
            counts[str(task.status)] = counts.get(str(task.status), 0) + 1
        return counts

    def compute_status(self) -> RunStatus:
        """Aggregate honestly.

        A run with one done task and one deferred task is NOT done -- collapsing
        it to ``done`` hides the deferral from whoever asked (spec 9).
        """
        if not self.tasks:
            return RunStatus.DONE
        counts = self.status_counts()
        total = len(self.tasks)
        if counts.get(str(TaskStatus.DONE), 0) == total:
            return RunStatus.DONE
        if counts.get(str(TaskStatus.DEFERRED), 0) == total:
            return RunStatus.DEFERRED
        if counts.get(str(TaskStatus.FAILED), 0) == total:
            return RunStatus.FAILED
        if counts.get(str(TaskStatus.AWAITING_REVIEW), 0) == total:
            return RunStatus.AWAITING_REVIEW
        return RunStatus.PARTIAL


class BibleEntry(BaseModel):
    """Something a model should know. Read by prompts."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _uid("b"))
    category: str
    text: str
    pinned: bool = False
    source_task: str | None = None
    created_at: datetime = Field(default_factory=_now)


class LedgerEntry(BaseModel):
    """Telemetry. Never injected into a prompt (spec 6)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _uid("l"))
    run_id: str = ""
    task_id: str = ""
    event: str = ""
    status: str = ""
    attempts: int = 0
    score: float | None = None
    judged: str | None = None
    judge_name: str | None = None
    producer: str | None = None
    model: str | None = None
    duration_s: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=_now)


class Bible(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[BibleEntry] = Field(default_factory=list)


class Ledger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[LedgerEntry] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _uid("p"))
    name: str
    engine: str
    bible: Bible = Field(default_factory=Bible)
    ledger: Ledger = Field(default_factory=Ledger)
    runs: list[Run] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def run(self, run_id: str) -> Run | None:
        for r in self.runs:
            if r.id == run_id:
                return r
        return None
