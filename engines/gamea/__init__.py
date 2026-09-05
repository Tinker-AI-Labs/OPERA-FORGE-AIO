"""
GAMEA — the 3D game-asset engine for OPERA.

Sibling to VIDEA/ARTISTA/MUSICA. Produces textured, cleaned-up 3D mesh
assets (.glb) from a text prompt or a folder of reference photos, judged
on objective mesh-quality criteria rather than opinion — closer to
FABRICA's "verifiable output" category than ARTISTA's aesthetic judging,
except the artifact is geometry instead of G-code/toolpaths.

Public surface, mirroring VIDEA/ARTISTA's engine module shape:
    GameaProducer  — engines/gamea/producer.py  (produce(brief) -> Artifact)
    MeshJudge      — engines/gamea/judge.py      (evaluate(task, artifact, context) -> Verdict)
    build          — engines/gamea/spec.py       (registers "gamea" with opera.registry)

Everything engine-specific (ComfyUI workflow selection, the local bridge
for Meshroom/Blender, mesh QC thresholds) lives under engines/gamea/.
Nothing in opera/ core knows GAMEA exists -- the same structural test that
tokenizes the core for VIDEA/ARTISTA/MUSICA role names covers "foundry" too
(tests/core/test_registry.py).
"""

from .clients import BridgeClient, ComfyUiClient, GameaClientError, GameaJobError
from .judge import MeshJudge
from .producer import GameaProducer
from .spec import KINDS, ROUTER_KEYWORDS, build, default_roles

__all__ = [
    "build", "default_roles", "KINDS", "ROUTER_KEYWORDS",
    "GameaProducer", "MeshJudge",
    "ComfyUiClient", "BridgeClient", "GameaClientError", "GameaJobError",
]
