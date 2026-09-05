"""
Objective mesh-quality checks — the thing that makes GAMEA's judge
verifiable rather than opinion-based (same reasoning the 2026-08-18
decision used to exclude FABRICA from the shared aesthetic-judge core:
fabrication output is objectively checkable, so is a mesh's).

Each function returns a plain (ok: bool, detail: dict) pair. MeshJudge
turns these into a weighted score — no single check hard-gates the
result, per the composition_intact lesson (2026-08-18): a hard gate that
one perception/measurement quirk gets wrong (like qwen2.5vl fabricating
corruption on a clean image) inverts the whole ranking. Weighting instead
of gating survived that lesson for vision judges; mesh checks are more
deterministic than vision-model perception, but the same defensive shape
still applies for the two checks that are heuristic (poly budget,
degenerate-face ratio) rather than binary facts (watertight, has UVs).
"""

from __future__ import annotations

from dataclasses import dataclass

import trimesh


@dataclass
class MeshQcResult:
    tri_count: int
    is_watertight: bool
    has_uv: bool
    degenerate_face_ratio: float
    bounding_box_extent: list


def load_mesh(path: str) -> trimesh.Trimesh:
    scene_or_mesh = trimesh.load(path, force="mesh")
    if isinstance(scene_or_mesh, trimesh.Scene):
        # Flatten multi-object scenes into one mesh for QC purposes —
        # matches blender_cleanup.py's own total-across-objects counting.
        geoms = list(scene_or_mesh.geometry.values())
        if not geoms:
            raise ValueError(f"no geometry found in {path}")
        scene_or_mesh = trimesh.util.concatenate(geoms)
    return scene_or_mesh


def inspect(path: str) -> MeshQcResult:
    mesh = load_mesh(path)

    tri_count = len(mesh.faces)
    is_watertight = bool(mesh.is_watertight)

    has_uv = bool(
        getattr(mesh.visual, "uv", None) is not None
        and len(getattr(mesh.visual, "uv", []) or []) > 0
    )

    # Degenerate faces: zero (or near-zero) area triangles left over from
    # AI-generated geometry or a too-aggressive decimate pass.
    areas = mesh.area_faces
    degenerate = (areas < 1e-8).sum() if len(areas) else 0
    degenerate_ratio = float(degenerate) / max(len(areas), 1)

    extent = mesh.bounding_box.extents.tolist() if len(mesh.vertices) else [0.0, 0.0, 0.0]

    return MeshQcResult(
        tri_count=tri_count,
        is_watertight=is_watertight,
        has_uv=has_uv,
        degenerate_face_ratio=degenerate_ratio,
        bounding_box_extent=extent,
    )
