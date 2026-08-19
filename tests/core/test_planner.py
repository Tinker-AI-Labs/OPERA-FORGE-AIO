"""Spec 4.5 / 5.1 / 5.2 -- planner output is validated, never trusted."""

import json

import pytest

from opera.config import RoleConfig
from opera.errors import PlannerError
from opera.llm.stub import StubLLMClient, think_wrapped
from opera.planner import LLMPlanner, Plan, PlannedTask, SingleTaskPlanner, validate_plan
from opera.registry import EngineSpec
from tests.doubles import ScriptedJudge, ScriptedProducer

ROLES = {
    "writer": RoleConfig(model="m", kind="text"),
    "coder": RoleConfig(model="m", kind="code"),
    "painter": RoleConfig(model="m", kind="image"),
}


def spec() -> EngineSpec:
    return EngineSpec(
        name="test", roles=ROLES,
        producers={"writer": ScriptedProducer(name="writer"),
                   "coder": ScriptedProducer(name="coder", kind="code"),
                   "painter": ScriptedProducer(name="painter", kind="image")},
        judge=ScriptedJudge([True]),
        kinds=frozenset({"text", "code", "image"}),
        default_role="writer", default_kind="text",
        router_keywords={"writer": ["write", "scene"], "coder": ["refactor", "function"],
                         "painter": ["concept art"]},
    )


def _client(payload) -> StubLLMClient:
    return StubLLMClient(default=payload if isinstance(payload, str) else json.dumps(payload))


# --- LLMPlanner --------------------------------------------------------------

async def test_planner_parses_a_task_list():
    client = _client({"tasks": [{"goal": "Write scene one", "role": "writer", "kind": "text"}],
                      "notes": ["keep it short"]})
    plan = await LLMPlanner(client, spec(), model="m").plan("a film")
    assert plan.tasks[0].goal == "Write scene one"
    assert plan.notes == ["keep it short"]


async def test_planner_requests_json_and_suppresses_reasoning():
    client = _client({"tasks": [{"goal": "g", "role": "writer"}]})
    await LLMPlanner(client, spec(), model="m").plan("a film")
    assert client.calls[0]["format_json"] is True
    assert client.calls[0]["no_think"] is True


async def test_planner_handles_think_wrapped_output():
    client = _client(think_wrapped({"tasks": [{"goal": "Write it", "role": "writer"}]}))
    plan = await LLMPlanner(client, spec(), model="m").plan("a film")
    assert plan.tasks[0].goal == "Write it"


async def test_planner_accepts_a_bare_list():
    client = _client([{"goal": "Write it", "role": "writer"}])
    assert len(( await LLMPlanner(client, spec(), model="m").plan("g")).tasks) == 1


async def test_planner_accepts_plain_strings_as_goals():
    client = _client({"tasks": ["Write scene one", "Write scene two"]})
    plan = await LLMPlanner(client, spec(), model="m").plan("g")
    assert [t.goal for t in plan.tasks] == ["Write scene one", "Write scene two"]
    assert plan.tasks[0].role is None


async def test_planner_truncates_to_max_tasks():
    client = _client({"tasks": [{"goal": f"g{i}"} for i in range(50)]})
    plan = await LLMPlanner(client, spec(), model="m", max_tasks=5).plan("g")
    assert len(plan.tasks) == 5


async def test_planner_prompt_lists_the_engine_vocabulary():
    client = _client({"tasks": [{"goal": "g"}]})
    await LLMPlanner(client, spec(), model="m").plan("a film", "MIRA IS THE KEEPER")
    prompt = client.calls[0]["prompt"]
    assert "writer" in prompt and "painter" in prompt
    assert "image" in prompt and "text" in prompt
    assert "MIRA IS THE KEEPER" in prompt


async def test_planner_raises_on_unparseable_output():
    with pytest.raises(PlannerError, match="unparseable"):
        await LLMPlanner(_client("here's my plan, roughly"), spec(), model="m").plan("g")


async def test_planner_raises_when_no_task_has_a_goal():
    with pytest.raises(PlannerError, match="no usable tasks"):
        await LLMPlanner(_client({"tasks": [{"role": "writer"}, {"goal": "  "}]}),
                         spec(), model="m").plan("g")


async def test_planner_raises_when_tasks_is_not_a_list():
    with pytest.raises(PlannerError, match="not a list"):
        await LLMPlanner(_client({"tasks": "just write it"}), spec(), model="m").plan("g")


def test_planner_requires_an_explicit_client():
    """Spec 5.4: no silent stub fallback."""
    with pytest.raises(PlannerError):
        LLMPlanner(None, spec(), model="m")


async def test_single_task_planner_is_the_honest_no_llm_default():
    plan = await SingleTaskPlanner(role="writer").plan("Write the whole film")
    assert len(plan.tasks) == 1 and plan.tasks[0].role == "writer"


# --- validate_plan (spec 4.5) ------------------------------------------------

def test_valid_plan_passes_through_unchanged():
    s = spec()
    tasks = validate_plan(Plan(tasks=[PlannedTask(goal="Write it", role="writer", kind="text")]),
                          s, s.router())
    assert tasks[0].role == "writer" and tasks[0].corrections == []


def test_hallucinated_role_is_routed_and_the_correction_is_recorded():
    s = spec()
    tasks = validate_plan(
        Plan(tasks=[PlannedTask(goal="Write the opening scene", role="cinematographer")]),
        s, s.router())
    assert tasks[0].role == "writer"
    assert "cinematographer" in tasks[0].corrections[0]


def test_missing_role_is_routed():
    s = spec()
    tasks = validate_plan(Plan(tasks=[PlannedTask(goal="refactor the parser function")]),
                          s, s.router())
    assert tasks[0].role == "coder"
    assert "no role given" in tasks[0].corrections[0]


def test_unknown_kind_falls_back_to_the_role_kind():
    s = spec()
    tasks = validate_plan(Plan(tasks=[PlannedTask(goal="Write it", role="writer", kind="hologram")]),
                          s, s.router())
    assert tasks[0].kind == "text"
    assert any("hologram" in c for c in tasks[0].corrections)


def test_kind_a_role_cannot_produce_is_corrected():
    s = spec()
    tasks = validate_plan(Plan(tasks=[PlannedTask(goal="Write it", role="writer", kind="image")]),
                          s, s.router())
    assert tasks[0].kind == "text"


def test_blank_goals_are_dropped():
    s = spec()
    tasks = validate_plan(Plan(tasks=[PlannedTask(goal="   ", role="writer"),
                                      PlannedTask(goal="Write it", role="writer")]),
                          s, s.router())
    assert len(tasks) == 1


def test_a_wholly_garbage_plan_still_yields_runnable_tasks():
    """Nothing the planner can emit should be able to sink the run at this stage."""
    s = spec()
    plan = Plan(tasks=[
        PlannedTask(goal="Write the opening scene", role="cinematographer", kind="hologram"),
        PlannedTask(goal="Refactor the render function", role="gaffer", kind="quantum"),
        PlannedTask(goal="Draw concept art of the tower", role="", kind=""),
    ])
    tasks = validate_plan(plan, s, s.router())
    assert [t.role for t in tasks] == ["writer", "coder", "painter"]
    assert [t.kind for t in tasks] == ["text", "code", "image"]
    assert all(t.corrections for t in tasks)


# --- role formatting slips are not hallucinations ----------------------------
# Found in a live run: llama3.2 emitted "Coder"/"Reasoner"/"Writer", and routing
# them by keyword sent a "Coder" task to the writer.

def test_title_cased_role_is_normalised_not_routed():
    s = spec()
    tasks = validate_plan(
        Plan(tasks=[PlannedTask(goal="Write the opening scene", role="Coder")]),
        s, s.router())
    assert tasks[0].role == "coder"          # not routed to "writer" on "Write"
    assert tasks[0].kind == "code"
    assert "normalised" in tasks[0].corrections[0]


def test_spaced_and_hyphenated_roles_are_normalised():
    roles = dict(ROLES, prompt_smith=RoleConfig(model="m", kind="text"))
    s = EngineSpec(
        name="test", roles=roles,
        producers={**{k: ScriptedProducer(name=k) for k in ROLES},
                   "prompt_smith": ScriptedProducer(name="prompt_smith")},
        judge=ScriptedJudge([True]), kinds=frozenset({"text", "code", "image"}),
        default_role="writer", default_kind="text",
    )
    for given in ("prompt smith", "Prompt-Smith", "PROMPT_SMITH"):
        tasks = validate_plan(Plan(tasks=[PlannedTask(goal="g", role=given)]), s, s.router())
        assert tasks[0].role == "prompt_smith", given


def test_a_genuinely_unknown_role_still_goes_to_the_router():
    s = spec()
    tasks = validate_plan(
        Plan(tasks=[PlannedTask(goal="refactor the parser function", role="Gaffer")]),
        s, s.router())
    assert tasks[0].role == "coder"
    assert "unknown role" in tasks[0].corrections[0]
