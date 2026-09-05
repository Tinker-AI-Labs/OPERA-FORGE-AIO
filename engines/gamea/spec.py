"""GAMEA EngineSpec.

Vocabulary: ``foundry`` -- GAMEA's own single production role, not a global
one. Not part of the drop-in tarball: it is the actual integration point
(the tarball's own README called this "wiring into opera/pipeline.py", but
there is no such file in the real source -- registering an EngineSpec via
opera/registry.py, exactly as ARTISTA/VIDEA/MUSICA's own spec.py files do,
is the real equivalent).

RECONCILED (checked against the real source, not assumed) -- router entry:
the tarball's README asked for "a case for GAMEA-shaped tasks" in
opera/router.py itself. The real opera/router.py is deliberately
engine-agnostic ("The vocabulary is engine-supplied. Nothing in this module
names a role.") -- it takes a keyword table from whichever EngineSpec built
it (see ARTISTA/VIDEA/MUSICA's own ROUTER_KEYWORDS below their own spec.py).
There is also no cross-engine content router anywhere in the real source:
which engine handles a request is chosen explicitly, by an ``--engine``
CLI flag or a project's ``engine`` field (service/context.py, service/cli.py)
-- not sniffed from prompt text. Hardcoding a "gamea" case into
opera/router.py would both contradict that module's own design and fail
tests/core/test_registry.py's structural guard (no engine's vocabulary may
appear as code in the core). The correct, real integration is this file:
GAMEA supplies its own ``ROUTER_KEYWORDS`` for its own ``foundry`` role,
the same way every other engine does, and registers itself below.
"""

from __future__ import annotations

from opera import registry
from opera.config import LLMConfig, RoleConfig
from opera.errors import ConfigError
from opera.protocols import LLMClient
from opera.registry import EngineSpec

from .clients import BridgeClient, ComfyUiClient
from .judge import MeshJudge, PASS_THRESHOLD
from .producer import GameaProducer

KINDS = frozenset({"model"})

PLANNER_SYSTEM = (
    "You are an asset director. Break the goal into a short ordered list of "
    "tasks, each one a single 3D asset to produce.\n"
    "Reply with JSON only, in this exact shape:\n"
    '{"tasks": [{"goal": "...", "role": "...", "kind": "..."}], "notes": ["..."]}\n'
    "Use only the roles and kinds the user lists."
)

# Multi-word or verb-led only, per router.py's fix 2 -- bare "model" or
# "asset" would match far too much ordinary language ("role model",
# "asset allocation").
ROUTER_KEYWORDS: dict[str, list[str]] = {
    "foundry": [
        "3d model", "3d asset", "game asset", "textured mesh", "glb model",
        "photogrammetry", "sculpt the asset", "generate the model",
        "render the model", "asset foundry",
    ],
}

DEFAULT_WORKFLOWS_DIR = "workflows"
DEFAULT_OUTPUT_DIR = "gamea_output"
DEFAULT_BRIDGE_URL = "http://127.0.0.1:5050"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"


def default_roles(llm: LLMConfig | None = None) -> dict[str, RoleConfig]:
    """GAMEA's role vocabulary.

    No LLM backs this role -- mesh generation goes through ComfyUI/Meshroom/
    Blender, and MeshJudge is deterministic -- so ``model`` is left empty
    rather than borrowing one of ``llm``'s names for a role that never calls
    it. Workflow/output directories and the two local service URLs are
    engine-specific settings threaded through ``options`` (RoleConfig's
    existing generic per-role settings dict) rather than new fields on the
    shared dataclass every other engine's roles use too (spec 3.3).
    """
    del llm  # accepted for the same signature every engine's default_roles() has
    return {
        "foundry": RoleConfig(
            model="",
            kind="model",
            max_attempts=1,
            pass_threshold=PASS_THRESHOLD,
            options={
                "workflows_dir": DEFAULT_WORKFLOWS_DIR,
                "output_dir": DEFAULT_OUTPUT_DIR,
                "bridge_url": DEFAULT_BRIDGE_URL,
                "comfy_url": DEFAULT_COMFY_URL,
            },
        ),
    }


def build(
    client: LLMClient,
    *,
    llm: LLMConfig | None = None,
    roles: dict[str, RoleConfig] | None = None,
    comfy_client: ComfyUiClient | None = None,
    bridge_client: BridgeClient | None = None,
) -> EngineSpec:
    # ``client`` is accepted, never used by this engine's own producer or
    # judge -- neither needs an LLM -- purely so gamea.build() matches the
    # same call convention service/context.py uses for every engine
    # (``registry.get(engine_name, client=..., llm=..., **kwargs)``).
    del client
    llm = llm or LLMConfig()
    roles = roles or default_roles(llm)
    if "foundry" not in roles:
        raise ConfigError("gamea needs a 'foundry' role")

    cfg = roles["foundry"]
    producer = GameaProducer(
        "foundry", cfg,
        comfy_client=comfy_client,
        bridge_client=bridge_client,
    )

    return EngineSpec(
        name="gamea",
        roles=roles,
        producers={"foundry": producer},
        judge=MeshJudge(pass_threshold=cfg.pass_threshold),
        kinds=KINDS,
        planner_system_prompt=PLANNER_SYSTEM,
        router_keywords=ROUTER_KEYWORDS,
        default_role="foundry",
        default_kind="model",
    )


registry.register("gamea", build)
