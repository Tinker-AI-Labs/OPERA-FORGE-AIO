"""Prompt -> task list, and the validation that makes planner output safe.

Spec 4.5: planner output is validated, never trusted. A model that hallucinates
``role="cinematographer"`` must not be able to sink a run -- the role is
corrected via the router, the correction is recorded on the task, and the run
continues.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .errors import JSONParseError, PlannerError
from .llm.parsing import extract_json
from .protocols import LLMClient
from .registry import EngineSpec
from .router import KeywordRouter
from .schemas import Task

DEFAULT_PLANNER_SYSTEM = (
    "You break a creative goal into a short ordered list of concrete tasks.\n"
    "Reply with JSON only, in this exact shape:\n"
    '{"tasks": [{"goal": "...", "role": "...", "kind": "..."}], "notes": ["..."]}\n'
    "Use only the roles and kinds the user lists. Keep the list as short as the "
    "goal allows. Each task goal must be self-contained and actionable."
)


class PlannedTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    goal: str
    role: str | None = None
    kind: str | None = None
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tasks: list[PlannedTask] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Planner(Protocol):
    async def plan(self, goal: str, context: str = "") -> Plan: ...


class LLMPlanner:
    """Asks a local model for a plan, in JSON, with reasoning traces suppressed."""

    name = "llm-planner"

    def __init__(
        self,
        client: LLMClient,
        spec: EngineSpec,
        *,
        model: str,
        timeout_s: float = 180.0,
        no_think: bool = True,
        max_tasks: int = 12,
    ) -> None:
        if client is None:
            # Spec 5.4: no silent stub fallback. A missing client is a wiring
            # mistake, and must present as one.
            raise PlannerError("LLMPlanner requires an explicit LLM client")
        self.client = client
        self.spec = spec
        self.model = model
        self.timeout_s = timeout_s
        self.no_think = no_think
        self.max_tasks = max_tasks

    def _system(self) -> str:
        return self.spec.planner_system_prompt or DEFAULT_PLANNER_SYSTEM

    def _user(self, goal: str, context: str) -> str:
        roles = ", ".join(sorted(self.spec.roles))
        kinds = ", ".join(sorted(self.spec.kinds))
        parts = [f"Available roles: {roles}", f"Available kinds: {kinds}"]
        if context.strip():
            parts.append(f"\nProject context:\n{context.strip()}")
        parts.append(f"\nGoal: {goal.strip()}")
        parts.append(f"\nReturn at most {self.max_tasks} tasks.")
        return "\n".join(parts)

    async def plan(self, goal: str, context: str = "") -> Plan:
        raw = await self.client.complete(
            system=self._system(),
            prompt=self._user(goal, context),
            model=self.model,
            format_json=True,   # spec 5.1
            timeout=self.timeout_s,
            no_think=self.no_think,  # spec 5.2
        )
        try:
            data = extract_json(raw, stage="planner")
        except JSONParseError as exc:
            raise PlannerError(f"planner returned unparseable output: {exc}") from exc

        if isinstance(data, list):
            data = {"tasks": data}
        if not isinstance(data, dict):
            raise PlannerError(f"planner returned {type(data).__name__}, expected an object")

        tasks_raw = data.get("tasks") or []
        if not isinstance(tasks_raw, list):
            raise PlannerError("planner 'tasks' was not a list")

        tasks: list[PlannedTask] = []
        for item in tasks_raw[: self.max_tasks]:
            if isinstance(item, str):
                tasks.append(PlannedTask(goal=item))
            elif isinstance(item, dict) and str(item.get("goal", "")).strip():
                tasks.append(PlannedTask(**{k: item.get(k) for k in ("goal", "role", "kind")
                                            if item.get(k) is not None}))
        if not tasks:
            raise PlannerError("planner produced no usable tasks")

        notes = [str(n) for n in (data.get("notes") or []) if str(n).strip()]
        return Plan(tasks=tasks, notes=notes)


class SingleTaskPlanner:
    """Treats the goal as one task. The honest default when no LLM is wanted."""

    name = "single-task"

    def __init__(self, role: str | None = None, kind: str | None = None) -> None:
        self.role = role
        self.kind = kind

    async def plan(self, goal: str, context: str = "") -> Plan:
        return Plan(tasks=[PlannedTask(goal=goal, role=self.role, kind=self.kind)])


def validate_plan(plan: Plan, spec: EngineSpec, router: KeywordRouter) -> list[Task]:
    """Turn a plan into tasks, correcting anything the engine cannot honour.

    Spec 4.5. Note that a stub planner emitting only valid roles can never
    exercise this -- the suite deliberately feeds it garbage.
    """
    tasks: list[Task] = []
    for planned in plan.tasks:
        goal = planned.goal.strip()
        if not goal:
            continue
        corrections: list[str] = []

        role = (planned.role or "").strip()
        if role and role not in spec.producers:
            # Models routinely title-case or space out a role they were given
            # verbatim ("Coder", "prompt smith"). That is a formatting slip, not
            # a hallucination -- resolve it directly rather than sending it to
            # the router, which would silently pick a different specialist.
            normalised = {r.lower().replace("_", " "): r for r in spec.producers}
            if match := normalised.get(role.lower().replace("_", " ").replace("-", " ")):
                corrections.append(f"role {role!r} normalised to {match!r}")
                role = match

        if role not in spec.producers:
            routed = router.route(goal)
            if role:
                corrections.append(
                    f"unknown role {role!r} -> {routed.role!r} ({routed.reason})"
                )
            else:
                corrections.append(f"no role given -> {routed.role!r} ({routed.reason})")
            role = routed.role

        kind = (planned.kind or "").strip()
        if kind not in spec.kinds:
            fallback = spec.kind_for_role(role)
            if kind:
                corrections.append(f"unknown kind {kind!r} -> {fallback!r}")
            kind = fallback

        # A planner may pair a valid role with a kind that role cannot make.
        role_kind = spec.kind_for_role(role)
        if kind != role_kind:
            corrections.append(f"kind {kind!r} not produced by role {role!r} -> {role_kind!r}")
            kind = role_kind

        tasks.append(Task(goal=goal, role=role, kind=kind, corrections=corrections))
    return tasks


def plan_to_dicts(plan: Plan) -> list[dict[str, Any]]:  # pragma: no cover - convenience
    return [t.model_dump() for t in plan.tasks]
