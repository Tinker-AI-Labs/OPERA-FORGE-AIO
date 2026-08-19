"""Orchestrates a full run over a planned task list.

The run-scoped rules from spec 4 live here:

* **4.3** context is recomputed at the top of every task iteration, so task 2
  can see task 1's output. Snapshotting it once per run defeats the entire
  point of project memory within a multi-scene run.
* **4.4** a failing task fails that task, not the run.
* **4.6** the project is persisted after every task, not once at the end.
* **4.7** no per-run state is assigned to ``self``. ``run()`` keeps everything
  in locals and passes it down explicitly, so two concurrent requests for
  different projects cannot cross-write each other's files.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import dataclass

from .bible import BibleWriter, LedgerWriter, ProjectStore
from .config import OperaConfig
from .errors import PlannerError
from .loop import execute_task
from .planner import Planner, SingleTaskPlanner, validate_plan
from .registry import EngineSpec
from .router import KeywordRouter
from .schemas import LedgerEntry, Project, Run, RunStatus, Task, TaskStatus


@dataclass(frozen=True)
class RunReport:
    run: Run
    project_id: str
    tasks_done: int
    tasks_failed: int
    tasks_deferred: int
    duration_s: float

    @property
    def status(self) -> RunStatus:
        return self.run.status


class Runner:
    """Stateless with respect to any individual run.

    Everything on ``self`` is configuration that outlives a run and is only
    read: the engine spec, the store, the planner, the router, the config.
    """

    def __init__(
        self,
        spec: EngineSpec,
        store: ProjectStore,
        *,
        planner: Planner | None = None,
        router: KeywordRouter | None = None,
        config: OperaConfig | None = None,
    ) -> None:
        self.spec = spec
        self.store = store
        self.config = config or OperaConfig()
        self.router = router or spec.router()
        self.planner = planner or SingleTaskPlanner()
        self.bible = BibleWriter(self.config)
        self.ledger = LedgerWriter(self.config)

    # -- public surface -------------------------------------------------------

    async def run_project(self, project_id: str, goal: str, *,
                          run_id: str | None = None) -> RunReport:
        """Load, run, and leave the project persisted on disk."""
        project = self.store.load(project_id)
        return await self.run(project, goal, run_id=run_id)

    async def run(self, project: Project, goal: str, *,
                  run_id: str | None = None) -> RunReport:
        """Execute one run against ``project``.

        Every mutable thing here is a local. Nothing is stashed on ``self``.
        """
        started = time.monotonic()
        # A caller may supply the id so it can report it before the run starts
        # (the API returns it immediately and the client polls -- spec 8).
        run = Run(project_id=project.id, engine=self.spec.name, goal=goal,
                  status=RunStatus.RUNNING, **({"id": run_id} if run_id else {}))
        project.runs.append(run)
        self.store.save(project)  # the run is visible to a poller immediately

        try:
            tasks = await self._plan(project, goal, run)
        except PlannerError as exc:
            run.status = RunStatus.FAILED
            run.error = f"planning failed: {exc}"
            run.finished_at = _utcnow()
            self.ledger.record(project.ledger, LedgerEntry(
                run_id=run.id, event="run_plan_failed", status=str(run.status),
                detail={"error": str(exc)}))
            self.store.save(project)
            return self._report(run, project, started)

        run.tasks = tasks
        self.store.save(project)  # 4.6: the plan itself survives a crash

        for task in run.tasks:
            await self._run_one(project, run, task)
            run.status = run.compute_status()
            self.store.save(project)  # 4.6: after each task, not at the end

        run.status = run.compute_status()
        run.finished_at = _utcnow()
        self.ledger.compact(project.ledger)
        self.store.save(project)
        return self._report(run, project, started)

    # -- internals ------------------------------------------------------------

    async def _plan(self, project: Project, goal: str, run: Run) -> list[Task]:
        context = self.bible.context(project.bible)
        plan = await self.planner.plan(goal, context)
        tasks = validate_plan(plan, self.spec, self.router)
        if not tasks:
            raise PlannerError("plan contained no usable tasks after validation")
        for note in plan.notes:
            # Planner notes are narrative, so they belong to the bible, not the
            # ledger -- but they are attributed, not laundered as fact.
            self.bible.add(project.bible, "brief", f"(plan note) {note}")
        self.bible.add(project.bible, "brief", goal.strip(), pinned=True)
        for task in tasks:
            if task.corrections:
                self.ledger.record(project.ledger, LedgerEntry(
                    run_id=run.id, task_id=task.id, event="plan_corrected",
                    status="pending", detail={"corrections": task.corrections}))
        return tasks

    async def _run_one(self, project: Project, run: Run, task: Task) -> None:
        started = time.monotonic()
        role_cfg = self.spec.role_config(task.role)

        # 4.3: recomputed here, inside the loop, so this task sees everything
        # earlier tasks in this same run established.
        context = self.bible.context(project.bible)

        try:
            producer = self.spec.producer_for(task.role)
            await execute_task(
                task,
                producer,
                self.spec.judge,
                context,
                config=self.config.loop,
                role_config=role_cfg,
                gate=self.spec.gate,
            )
        except Exception as exc:  # noqa: BLE001
            # 4.4: contain the blast radius. A planner hallucinating a role, or
            # a producer crashing, must not discard four completed scenes.
            task.status = TaskStatus.FAILED
            task.error = f"{type(exc).__name__}: {exc}"
            task.finished_at = _utcnow()

        if task.status is TaskStatus.DONE and task.artifact is not None:
            self.bible.record_artifact(project.bible, task, task.artifact)

        self.ledger.record_task(
            project.ledger, run, task,
            duration_s=time.monotonic() - started,
            model=role_cfg.model if role_cfg else None,
        )

    def _report(self, run: Run, project: Project, started: float) -> RunReport:
        counts = run.status_counts()
        return RunReport(
            run=run,
            project_id=project.id,
            tasks_done=counts.get(str(TaskStatus.DONE), 0),
            tasks_failed=counts.get(str(TaskStatus.FAILED), 0),
            tasks_deferred=counts.get(str(TaskStatus.DEFERRED), 0),
            duration_s=time.monotonic() - started,
        )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
