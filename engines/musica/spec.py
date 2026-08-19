"""MUSICA EngineSpec.

Vocabulary: ``composer / arranger / mixer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opera import registry
from opera.config import LLMConfig, RoleConfig
from opera.errors import ConfigError
from opera.protocols import HumanGate, LLMClient
from opera.registry import EngineSpec

from .gate import HoldForHuman
from .judge import KeyTempoDetector, MusicSpec, build_judge
from .producers import ARRANGER_SYSTEM, AceStepProducer, LLMPlanProducer

KINDS = frozenset({"text", "audio"})

PLANNER_SYSTEM = (
    "You are a music producer. Break the goal into a short ordered list of tasks: "
    "composition and arrangement come before any render.\n"
    "Reply with JSON only, in this exact shape:\n"
    '{"tasks": [{"goal": "...", "role": "...", "kind": "..."}], "notes": ["..."]}\n'
    "Use only the roles and kinds the user lists."
)

ROUTER_KEYWORDS: dict[str, list[str]] = {
    "composer": ["write the theme", "compose the", "chord progression", "melody for",
                 "main theme", "motif"],
    "arranger": ["arrange the", "arrangement for", "instrumentation", "orchestrate",
                 "voicing", "section breakdown"],
    # Multi-word or verb-led only -- bare "music" or "score" would match
    # "a musical score plays in the background".
    "mixer": ["render the audio", "background score", "musical cue", "mix the track",
              "master the track", "bounce the stems"],
}


def default_roles(llm: LLMConfig | None = None) -> dict[str, RoleConfig]:
    llm = llm or LLMConfig()
    return {
        "composer": RoleConfig(model=llm.default_model, kind="text", temperature=0.85,
                               max_attempts=2, no_think=True),
        "arranger": RoleConfig(model=llm.default_model, kind="text", temperature=0.5,
                               max_attempts=2, no_think=True,
                               system_prompt=ARRANGER_SYSTEM),
        "mixer": RoleConfig(model=llm.default_model, kind="audio", max_attempts=2,
                            timeout_s=900.0),
    }


def build(
    client: LLMClient,
    *,
    llm: LLMConfig | None = None,
    roles: dict[str, RoleConfig] | None = None,
    output_dir: str | Path = "./musica_out",
    audio_producer: Any = None,
    music_spec: MusicSpec | None = None,
    detector: KeyTempoDetector | None = None,
    gate: HumanGate | None = None,
    require_human: bool = True,
    threshold: float = 0.7,
) -> EngineSpec:
    llm = llm or LLMConfig()
    roles = roles or default_roles(llm)
    for required in ("composer", "arranger", "mixer"):
        if required not in roles:
            raise ConfigError(f"musica needs a {required!r} role")

    if gate is None and require_human:
        gate = HoldForHuman()

    producers: dict[str, Any] = {
        "composer": LLMPlanProducer(client, "composer", roles["composer"]),
        "arranger": LLMPlanProducer(client, "arranger", roles["arranger"],
                                    system=ARRANGER_SYSTEM),
        "mixer": audio_producer or AceStepProducer("mixer", roles["mixer"],
                                                   output_dir=output_dir),
    }

    return EngineSpec(
        name="musica",
        roles=roles,
        producers=producers,
        judge=build_judge(client, model=llm.judge_model, spec=music_spec,
                          detector=detector, threshold=threshold),
        kinds=KINDS,
        planner_system_prompt=PLANNER_SYSTEM,
        router_keywords=ROUTER_KEYWORDS,
        default_role="composer",
        default_kind="text",
        media_kinds=frozenset({"audio"}),
        gate=gate,
    )


registry.register("musica", build)
