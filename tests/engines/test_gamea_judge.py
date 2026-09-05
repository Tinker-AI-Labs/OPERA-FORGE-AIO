"""
MeshJudge tests against synthetic trimesh primitives — no Blender,
Meshroom, or ComfyUI required, same "stub-only, no external tool
required" bar VIDEA's 8/8 and OPERA's 327/327 held to.

Run: pytest tests/engines/test_gamea_judge.py -v
"""

import os
import tempfile

import trimesh
import pytest

from engines.gamea.judge import MeshJudge
from opera.schemas import Artifact

# RECONCILED (checked against the real source, not assumed): the tarball's
# FakeArtifact stand-in carried a `file_path` attribute that doesn't exist on
# opera/schemas.py's real Artifact (its field is `path`). Real Artifact is a
# plain pydantic model that's trivial to construct directly, so it is used
# here instead of a hand-rolled stand-in.


def _save_temp_glb(mesh: trimesh.Trimesh) -> str:
    fd, path = tempfile.mkstemp(suffix=".glb")
    os.close(fd)
    mesh.export(path)
    return path


def test_clean_watertight_box_passes():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    path = _save_temp_glb(mesh)
    verdict = MeshJudge().judge(Artifact(path=path))
    assert verdict.passed, verdict.issues
    assert verdict.score > 0.7


def test_unreadable_file_hard_fails():
    fd, path = tempfile.mkstemp(suffix=".glb")
    os.write(fd, b"not a real glb file")
    os.close(fd)
    verdict = MeshJudge().judge(Artifact(path=path))
    assert verdict.passed is False
    assert verdict.score == 0.0
    assert "unreadable" in verdict.issues[0]


def test_oversized_mesh_scores_lower_but_not_zero():
    # subdivide a sphere repeatedly to blow well past the tri budget
    mesh = trimesh.creation.icosphere(subdivisions=6)
    path = _save_temp_glb(mesh)
    verdict = MeshJudge().judge(Artifact(path=path))
    assert any("tri budget" in i for i in verdict.issues)
    assert 0.0 < verdict.score < 1.0


def test_passed_is_score_backed_not_trusted_from_input():
    """The standing rule from the 2026-08-18 defect fixes: passed must
    come from the judge's own score against threshold, never an upstream
    claim. Assert a judge instantiated with an impossible threshold
    can never pass, regardless of mesh quality."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    path = _save_temp_glb(mesh)
    verdict = MeshJudge(pass_threshold=1.01).judge(Artifact(path=path))
    assert verdict.passed is False


def test_degenerate_bounding_box_hard_fails():
    # a mesh with (near) zero extent in every axis
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0]] * 3, faces=[[0, 1, 2]])
    path = _save_temp_glb(mesh)
    verdict = MeshJudge().judge(Artifact(path=path))
    assert verdict.passed is False
    assert verdict.score == 0.0
