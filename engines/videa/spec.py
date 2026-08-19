"""VIDEA EngineSpec.

``writer / reasoner / coder`` is VIDEA's own text vocabulary -- it is defined
here, in config, not in the core (spec 3.3).
"""

from __future__ import annotations

from opera import registry
from opera.config import LLMConfig, RoleConfig
from opera.errors import ConfigError
from opera.protocols import LLMClient
from opera.registry import EngineSpec

from .judge import build_judge
from .producers import LLMProducer

KINDS = frozenset({"text", "code", "video"})

PLANNER_SYSTEM = (
    "You are the producer on a film project. Break the goal into a short ordered "
    "list of concrete tasks, each one thing a single specialist can complete.\n"
    "Reply with JSON only, in this exact shape:\n"
    '{"tasks": [{"goal": "...", "role": "...", "kind": "..."}], "notes": ["..."]}\n'
    "Use only the roles and kinds the user lists. Prefer fewer, larger tasks over "
    "many tiny ones. Each task goal must stand on its own without the others."
)

ROUTER_KEYWORDS: dict[str, list[str]] = {
    "writer": ["write", "script", "screenplay", "scene", "dialogue", "monologue",
               "voiceover", "narration", "logline", "synopsis", "treatment"],
    "reasoner": ["analyse", "analyze", "evaluate", "compare", "critique", "plan",
                 "outline", "structure", "pacing", "continuity"],
    "coder": ["code", "function", "script file", "refactor", "implement", "automate",
              "pipeline", "ffmpeg", "batch"],
    # Media keywords are multi-word or verb-led, per the spec 7 fix. Bare "shot"
    # or "clip" would match far too much ordinary film talk.
    "editor": ["render the video", "video clip", "cut the sequence", "assemble the edit",
               "rough cut", "final cut"],
}


def default_roles(llm: LLMConfig | None = None) -> dict[str, RoleConfig]:
    """VIDEA's role vocabulary. Models come from config, never hardcoded here."""
    llm = llm or LLMConfig()
    return {
        "writer": RoleConfig(
            model=llm.default_model,
            kind="text",
            temperature=0.85,
            max_attempts=2,
            no_think=True,  # the trace is not the product here
            system_prompt=(
                "You are a screenwriter. Write vivid, economical prose that honours "
                "the established characters, facts and style in the project context. "
                "Output only the work itself -- no preamble, no commentary."
            ),
        ),
        "reasoner": RoleConfig(
            model=llm.default_model,
            kind="text",
            temperature=0.3,
            max_attempts=2,
            no_think=False,  # this is the one role where the reasoning is the point
            system_prompt=(
                "You are a story editor. Analyse structure, pacing and continuity. "
                "Be concrete and specific. Output only the analysis."
            ),
        ),
        "coder": RoleConfig(
            model=llm.default_model,
            kind="code",
            temperature=0.2,
            max_attempts=2,
            no_think=True,
            system_prompt=(
                "You are a pipeline engineer. Write correct, runnable code with no "
                "placeholder sections. Output only code, in a single block."
            ),
        ),
    }


def build(
    client: LLMClient,
    *,
    llm: LLMConfig | None = None,
    roles: dict[str, RoleConfig] | None = None,
    default_role: str | None = None,
    threshold: float = 0.7,
    with_video_judge: bool = True,
) -> EngineSpec:
    llm = llm or LLMConfig()
    roles = roles or default_roles(llm)
    producers = {name: LLMProducer(client, name, cfg) for name, cfg in roles.items()}
    # A caller supplying its own vocabulary must not inherit this module's
    # assumption that "writer" exists.
    if default_role is None:
        default_role = "writer" if "writer" in roles else sorted(roles)[0]
    elif default_role not in roles:
        raise ConfigError(f"videa default_role {default_role!r} is not in the supplied roles")
    default_kind = roles[default_role].kind
    return EngineSpec(
        name="videa",
        roles=roles,
        producers=producers,
        judge=build_judge(client, model=llm.judge_model, vision_model=llm.vision_model,
                          threshold=threshold, with_video=with_video_judge),
        kinds=KINDS | {cfg.kind for cfg in roles.values()},
        planner_system_prompt=PLANNER_SYSTEM,
        router_keywords={k: v for k, v in ROUTER_KEYWORDS.items() if k in roles},
        default_role=default_role,
        default_kind=default_kind,
    )


registry.register("videa", build)
