"""OPERA command line.

``run``, ``health``, ``serve``, ``projects``, ``show``. ``--engine`` selects the
EngineSpec; ``--stub`` is the explicit opt-in to stub client and stub output
(spec 8) -- it is never chosen for you.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Sequence

from opera import registry
from opera.bible import BibleWriter
from opera.errors import OperaError
from opera.schemas import RunStatus, TaskStatus

from .context import build_context

STATUS_MARK = {
    TaskStatus.DONE: "ok",
    TaskStatus.FAILED: "FAIL",
    TaskStatus.DEFERRED: "deferred",
    TaskStatus.AWAITING_REVIEW: "awaiting review",
    TaskStatus.PENDING: "pending",
    TaskStatus.RUNNING: "running",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opera", description=__doc__.splitlines()[0])
    parser.add_argument("--engine", default="videa", help="engine spec to use")
    parser.add_argument("--stub", action="store_true",
                        help="use the stub client and stub producers (no model host)")
    parser.add_argument("--config", default=None, help="path to a JSON config file")
    parser.add_argument("--projects-dir", default=None, help="override the project store root")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="emit machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="plan and execute a goal")
    p_run.add_argument("goal")
    p_run.add_argument("--project", default=None, help="existing project id")
    p_run.add_argument("--name", default=None, help="name for a new project")
    p_run.add_argument(
        "--param", action="append", default=[], metavar="KEY=VALUE",
        help="set a param on every planned task (repeatable); read by a "
             "producer through Brief.params, e.g. --param images_dir=./photos "
             "for GAMEA's photo-folder path, or --param prompt='...' for "
             "ARTISTA's explicit prompt override",
    )

    sub.add_parser("health", help="report engine and model-host reachability")

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("projects", help="list projects")

    p_show = sub.add_parser("show", help="show a run in detail")
    p_show.add_argument("run_id")
    p_show.add_argument("--project", required=True)

    p_bible = sub.add_parser("bible", help="print a project's rendered context")
    p_bible.add_argument("--project", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_dispatch(args))
    except OperaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


async def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "serve":
        return _serve(args)

    ctx = build_context(engine=args.engine, stub=args.stub,
                        config_path=args.config, projects_dir=args.projects_dir)
    try:
        handler = {
            "run": _cmd_run,
            "health": _cmd_health,
            "projects": _cmd_projects,
            "show": _cmd_show,
            "bible": _cmd_bible,
        }[args.command]
        return await handler(ctx, args)
    finally:
        await ctx.aclose()


async def _cmd_run(ctx, args) -> int:
    if args.project:
        project = ctx.store.load(args.project)
    else:
        project = ctx.store.create(args.name or args.goal[:60], ctx.engine_name)

    task_params = _parse_params(args.param)
    report = await ctx.runner().run(project, args.goal, task_params=task_params)
    run = report.run

    if args.as_json:
        print(json.dumps(_run_payload(run), indent=2))
        return 0 if run.status is RunStatus.DONE else 1

    print(f"project {project.id}  run {run.id}  [{run.status}]")
    if run.error:
        print(f"  error: {run.error}")
    for task in run.tasks:
        mark = STATUS_MARK.get(task.status, str(task.status))
        print(f"  - [{mark}] {task.role}/{task.kind}: {task.goal}")
        for correction in task.corrections:
            print(f"      corrected: {correction}")
        if task.verdict:
            print(f"      score {task.verdict.score:.2f} "
                  f"(judged: {task.verdict.judged}, by {task.verdict.judge_name})")
            for issue in task.verdict.issues[:3]:
                print(f"      issue: {issue}")
        if task.deferred_reason:
            print(f"      deferred: {task.deferred_reason}")
        if task.error:
            print(f"      error: {task.error}")
        if task.artifact and task.artifact.path:
            print(f"      file: {task.artifact.path}")
    print(f"\n{report.tasks_done} done, {report.tasks_failed} failed, "
          f"{report.tasks_deferred} deferred in {report.duration_s:.1f}s")
    return 0 if run.status is RunStatus.DONE else 1


async def _cmd_health(ctx, args) -> int:
    reachable = await ctx.client.available()
    spec = ctx.spec()
    payload: dict[str, Any] = {
        "engine": spec.name,
        "engines_available": registry.available(),
        "llm_client": getattr(ctx.client, "name", "?"),
        "llm_reachable": reachable,
        "stub": ctx.stub,
        "projects_dir": str(ctx.config.projects_dir),
        "producers": {role: {"kind": p.kind, "available": bool(getattr(p, "available", False))}
                      for role, p in spec.producers.items()},
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"engine        {payload['engine']}")
        print(f"registered    {', '.join(payload['engines_available'])}")
        print(f"llm           {payload['llm_client']} "
              f"({'reachable' if reachable else 'UNREACHABLE'})"
              f"{' [stub]' if ctx.stub else ''}")
        print(f"projects      {payload['projects_dir']}")
        print("producers")
        for role, info in payload["producers"].items():
            state = "available" if info["available"] else "UNAVAILABLE (tasks will defer)"
            print(f"  {role:<14} {info['kind']:<7} {state}")
    return 0 if (reachable or ctx.stub) else 1


async def _cmd_projects(ctx, args) -> int:
    rows = ctx.store.list_projects()
    if args.as_json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no projects")
        return 0
    for row in rows:
        print(f"{row['id']}  {row['engine']:<10} {row['name']}")
    return 0


async def _cmd_show(ctx, args) -> int:
    project = ctx.store.load(args.project)
    run = project.run(args.run_id)
    if run is None:
        print(f"error: no run {args.run_id} in project {args.project}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(_run_payload(run), indent=2))
        return 0
    print(f"run {run.id}  [{run.status}]  {run.goal}")
    for task in run.tasks:
        print(f"\n[{STATUS_MARK.get(task.status, task.status)}] {task.role}/{task.kind}: {task.goal}")
        if task.artifact:
            print(task.artifact.excerpt(1200))
    return 0


async def _cmd_bible(ctx, args) -> int:
    project = ctx.store.load(args.project)
    print(BibleWriter(ctx.config).context(project.bible))
    return 0


def _parse_params(raw: list[str]) -> dict[str, str]:
    """Parse repeated ``--param KEY=VALUE`` flags into a dict.

    Values are always strings -- a producer that expects something else
    (an int, say) is responsible for its own conversion, same as any other
    CLI-sourced string.
    """
    params: dict[str, str] = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep:
            raise OperaError(f"--param must be KEY=VALUE, got {item!r}")
        params[key.strip()] = value
    return params


def _run_payload(run) -> dict[str, Any]:
    return {
        "id": run.id,
        "goal": run.goal,
        "status": str(run.status),
        "counts": run.status_counts(),
        "error": run.error,
        "tasks": [
            {
                "id": t.id, "goal": t.goal, "role": t.role, "kind": t.kind,
                "status": str(t.status), "attempts": t.attempts,
                "corrections": t.corrections,
                "deferred_reason": t.deferred_reason, "error": t.error,
                "verdict": t.verdict.model_dump(mode="json") if t.verdict else None,
                "path": t.artifact.path if t.artifact else None,
            }
            for t in run.tasks
        ],
    }


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("error: uvicorn is not installed; pip install 'opera[service]'", file=sys.stderr)
        return 1
    from .api import create_app

    app = create_app(engine=args.engine, stub=args.stub,
                     config_path=args.config, projects_dir=args.projects_dir)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
