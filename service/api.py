"""FastAPI surface.

Runs take minutes. Submission returns a run id immediately and status is polled
(spec 8) -- no HTTP connection is held open across a render. Lifespan handlers,
not the deprecated ``@app.on_event``.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from opera import registry
from opera.bible import BibleWriter
from opera.errors import ProjectStoreError
from opera.schemas import RunStatus

from .context import AppContext, build_context


class CreateProject(BaseModel):
    name: str
    engine: str | None = None


class SubmitRun(BaseModel):
    goal: str = Field(min_length=1)
    # Applied to every planned task (Task.params -> Brief.params), same
    # override the CLI's repeatable --param KEY=VALUE gives -- e.g. GAMEA's
    # images_dir/workflow, or ARTISTA's explicit prompt override. Engine-
    # supplied vocabulary; the API itself never interprets these keys.
    params: dict[str, str] = Field(default_factory=dict)


class RunAccepted(BaseModel):
    run_id: str
    project_id: str
    status: str
    poll: str


def create_app(context: AppContext | None = None, **ctx_kwargs: Any) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ctx = context or build_context(**ctx_kwargs)
        # Tracked so shutdown can wait rather than abandoning work mid-render.
        app.state.tasks = set()
        try:
            yield
        finally:
            for task in list(app.state.tasks):
                task.cancel()
            if app.state.tasks:
                await asyncio.gather(*app.state.tasks, return_exceptions=True)
            await app.state.ctx.aclose()

    app = FastAPI(title="OPERA", version="0.1.0", lifespan=lifespan)

    def ctx(request: Request) -> AppContext:
        return request.app.state.ctx

    @app.get("/health")
    async def health(c: AppContext = Depends(ctx)) -> dict[str, Any]:
        reachable = await c.client.available()
        spec = c.spec()
        return {
            "status": "ok" if (reachable or c.stub) else "degraded",
            "engine": spec.name,
            "engines_available": registry.available(),
            "llm": {"client": getattr(c.client, "name", "?"), "reachable": reachable,
                    "stub": c.stub},
            "producers": {
                role: {"kind": p.kind, "available": bool(getattr(p, "available", False))}
                for role, p in spec.producers.items()
            },
            "projects_dir": str(c.config.projects_dir),
        }

    @app.get("/projects")
    async def list_projects(c: AppContext = Depends(ctx)) -> list[dict[str, str]]:
        return c.store.list_projects()

    @app.post("/projects", status_code=201)
    async def create_project(body: CreateProject, c: AppContext = Depends(ctx)) -> dict[str, str]:
        project = c.store.create(body.name, body.engine or c.engine_name)
        return {"id": project.id, "name": project.name, "engine": project.engine}

    @app.get("/projects/{project_id}")
    async def get_project(project_id: str, c: AppContext = Depends(ctx)) -> dict[str, Any]:
        project = _load(c, project_id)
        return {
            "id": project.id, "name": project.name, "engine": project.engine,
            "updated_at": project.updated_at.isoformat(),
            "runs": [{"id": r.id, "goal": r.goal, "status": str(r.status),
                      "tasks": len(r.tasks)} for r in project.runs],
        }

    @app.get("/projects/{project_id}/bible")
    async def read_bible(project_id: str, c: AppContext = Depends(ctx)) -> dict[str, Any]:
        project = _load(c, project_id)
        writer = BibleWriter(c.config)
        return {
            "entries": [e.model_dump(mode="json") for e in project.bible.entries],
            "context": writer.context(project.bible),
        }

    @app.post("/projects/{project_id}/runs", status_code=202,
              response_model=RunAccepted)
    async def submit_run(project_id: str, body: SubmitRun,
                         request: Request, c: AppContext = Depends(ctx)) -> RunAccepted:
        project = _load(c, project_id)
        runner = c.runner()
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        async def execute() -> None:
            try:
                await runner.run_project(project_id, body.goal, run_id=run_id,
                                         task_params=body.params or None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                # The runner records per-task failures itself; reaching here
                # means something outside the task loop broke. Record it on the
                # project so a poller sees a reason rather than a run that never
                # finishes.
                _mark_crashed(c, project_id, run_id, exc)

        task = asyncio.create_task(execute())
        request.app.state.tasks.add(task)
        task.add_done_callback(request.app.state.tasks.discard)

        return RunAccepted(
            run_id=run_id,
            project_id=project.id,
            status=str(RunStatus.PENDING),
            poll=f"/projects/{project.id}/runs/{run_id}",
        )

    @app.get("/projects/{project_id}/runs")
    async def list_runs(project_id: str, c: AppContext = Depends(ctx)) -> list[dict[str, Any]]:
        project = _load(c, project_id)
        return [_run_summary(r) for r in project.runs]

    @app.get("/projects/{project_id}/runs/{run_id}")
    async def get_run(project_id: str, run_id: str, c: AppContext = Depends(ctx)) -> dict[str, Any]:
        project = _load(c, project_id)
        run = project.run(run_id)
        if run is None:
            raise HTTPException(404, f"no run {run_id} in project {project_id}")
        detail = _run_summary(run)
        detail["tasks"] = [
            {
                "id": t.id, "goal": t.goal, "role": t.role, "kind": t.kind,
                "status": str(t.status), "attempts": t.attempts,
                "error": t.error, "deferred_reason": t.deferred_reason,
                "corrections": t.corrections,
                "verdict": t.verdict.model_dump(mode="json") if t.verdict else None,
                "artifact": (
                    {"id": t.artifact.id, "kind": t.artifact.kind, "path": t.artifact.path,
                     "excerpt": t.artifact.excerpt()} if t.artifact else None
                ),
            }
            for t in run.tasks
        ]
        return detail

    def _load(c: AppContext, project_id: str):
        try:
            return c.store.load(project_id)
        except ProjectStoreError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app


def _mark_crashed(c: AppContext, project_id: str, run_id: str, exc: Exception) -> None:
    try:
        project = c.store.load(project_id)
    except ProjectStoreError:
        return
    run = project.run(run_id)
    if run is None:
        return
    run.status = RunStatus.FAILED
    run.error = f"{type(exc).__name__}: {exc}"
    try:
        c.store.save(project)
    except ProjectStoreError:
        pass


def _run_summary(run) -> dict[str, Any]:
    return {
        "id": run.id,
        "goal": run.goal,
        "engine": run.engine,
        "status": str(run.status),
        "counts": run.status_counts(),
        "error": run.error,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }
