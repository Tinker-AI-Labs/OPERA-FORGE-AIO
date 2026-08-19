"""ARTISTA EngineSpec.

Vocabulary: ``concept / prompt_smith / retoucher`` -- ARTISTA's own, not a
global one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opera import registry
from opera.config import LLMConfig, RoleConfig
from opera.errors import ConfigError
from opera.protocols import LLMClient
from opera.registry import EngineSpec

from .judge import build_judge
from .producers import ComfyUIProducer, PromptSmith

KINDS = frozenset({"text", "image"})

PLANNER_SYSTEM = (
    "You are an art director. Break the goal into a short ordered list of tasks. "
    "Prompt-writing and image generation are separate tasks -- write the prompt "
    "first, then generate from it.\n"
    "Reply with JSON only, in this exact shape:\n"
    '{"tasks": [{"goal": "...", "role": "...", "kind": "..."}], "notes": ["..."]}\n'
    "Use only the roles and kinds the user lists."
)

ROUTER_KEYWORDS: dict[str, list[str]] = {
    "concept": ["mood board", "art direction", "visual language", "look and feel",
                "colour script", "color script", "describe the look"],
    "prompt_smith": ["write the prompt", "image prompt", "generation prompt",
                     "prompt for"],
    # Multi-word or verb-led only -- bare "art" would match "state of the art".
    "retoucher": ["concept art", "key frame", "key art", "generate the image",
                  "render the image", "illustration", "poster", "cover art"],
}


def default_roles(llm: LLMConfig | None = None) -> dict[str, RoleConfig]:
    llm = llm or LLMConfig()
    return {
        "concept": RoleConfig(
            model=llm.default_model, kind="text", temperature=0.9, max_attempts=2,
            no_think=True,
            system_prompt=(
                "You are a concept artist working in words. Describe the visual "
                "direction concretely: subject, composition, lighting, palette, "
                "medium. Output only the description."
            ),
        ),
        "prompt_smith": RoleConfig(
            model=llm.default_model, kind="text", temperature=0.6, max_attempts=2,
            no_think=True,
        ),
        "retoucher": RoleConfig(
            model=llm.vision_model, kind="image", max_attempts=2, timeout_s=600.0,
            # 2026-08-19: VisionJudge's rubric ranks images correctly but
            # rarely clears an absolute pass_threshold even on genuinely good
            # work (measured ~0.567-0.7 against a real vision model) --
            # threshold-gating would fail everything. best_of_n keeps the
            # ranking useful without needing it to double as a pass bar.
            # ceiling=0.85 is a real "good enough, don't render again" bar
            # given how expensive a render is on this hardware (measured
            # several minutes each, see docs/ARTISTA_HARDWARE_FINDINGS.md) --
            # an attempt that clearly good stops the batch early.
            policy="best_of_n", ceiling=0.85,
        ),
    }


def build(
    client: LLMClient,
    *,
    llm: LLMConfig | None = None,
    roles: dict[str, RoleConfig] | None = None,
    comfyui_host: str = "http://127.0.0.1:8188",
    workflow: dict[str, Any] | None = None,
    workflow_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    image_producer: Any = None,
    threshold: float = 0.7,
) -> EngineSpec:
    llm = llm or LLMConfig()
    roles = roles or default_roles(llm)

    for required in ("concept", "prompt_smith", "retoucher"):
        if required not in roles:
            raise ConfigError(f"artista needs a {required!r} role")

    producers: dict[str, Any] = {
        "concept": PromptSmith(client, "concept", roles["concept"]),
        "prompt_smith": PromptSmith(client, "prompt_smith", roles["prompt_smith"]),
        "retoucher": image_producer or ComfyUIProducer(
            "retoucher", roles["retoucher"], host=comfyui_host,
            workflow=workflow, workflow_path=workflow_path, output_dir=output_dir,
        ),
    }

    return EngineSpec(
        name="artista",
        roles=roles,
        producers=producers,
        judge=build_judge(client, vision_model=llm.vision_model,
                          text_model=llm.default_model, threshold=threshold),
        kinds=KINDS,
        planner_system_prompt=PLANNER_SYSTEM,
        router_keywords=ROUTER_KEYWORDS,
        default_role="concept",
        default_kind="text",
    )


registry.register("artista", build)
