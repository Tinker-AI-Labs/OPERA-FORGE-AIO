"""
MeshJudge -- judges a GAMEA artifact's mesh quality.

Follows the two standing rules the 2026-08-18 defect fixes established
for every judge in this codebase (test_judge_authority.py, 21 tests):

  1. `passed` is set BY THE JUDGE from the score against threshold, never
     read off some upstream flag it doesn't control. There's no model
     output to mistrust here (no LLM in the loop for mesh QC), but the
     same discipline applies to blender_cleanup.py's own reported stats
     -- this judge re-inspects the file itself with mesh_qc rather than
     trusting RESULT_JSON's self-reported tris_after number.
  2. Weighted rubric, not a hard gate on any single field -- the
     composition_intact lesson (weighting fixed a ranking inversion that
     gating caused). Only exception: a completely unreadable/corrupt file
     is a hard fail, because there's no partial credit for "doesn't
     parse" the way there can be for "parses but is a bit dense."

RECONCILED (checked against the real source, not assumed): opera/schemas.py's
real ``Verdict`` is a pydantic model with ``judged`` and ``judge_name`` as
required fields (``extra="forbid"``) -- there is no ``judged_by``, which the
tarball's stand-in used. ``judged="artifact"`` is correct here (unlike a
judge that only ever saw a plan): mesh_qc.inspect() re-reads the actual
produced bytes on disk, not a description of them.

opera/protocols.py's ``Judge`` protocol is ``async def evaluate(self, task,
artifact, context) -> Verdict`` -- that is the method opera/loop.py actually
calls, so it is the one that has to exist for MeshJudge to function as a
real Judge. A synchronous ``judge(artifact)`` convenience method is kept
alongside it (mesh QC needs no event loop of its own) for standalone/manual
use, and is what tests/engines/test_gamea_judge.py's synthetic-mesh tests
exercise directly without needing a Task or an event loop.
"""

from __future__ import annotations

from opera.schemas import Artifact, Task, Verdict

from .mesh_qc import inspect, MeshQcResult

# Defaults -- tune per-asset like blender_cleanup.py's REMESH_VOXEL_SIZE
# is documented as a guess to tune, not a fixed truth.
TRI_BUDGET_MAX = 30_000
TRI_BUDGET_WARN = 20_000
MAX_DEGENERATE_RATIO = 0.02
PASS_THRESHOLD = 0.70


class MeshJudge:
    """judge(artifact) / evaluate(task, artifact, context) -> Verdict,
    matching the Producer/Judge shape ARTISTA and VIDEA already use so
    opera/loop.py's produce -> judge -> revise and best_of_n policy work
    unmodified."""

    name = "gamea-mesh-judge"

    def __init__(self, pass_threshold: float = PASS_THRESHOLD):
        self.pass_threshold = pass_threshold

    def judge(self, artifact: Artifact, goal: str | None = None) -> Verdict:
        issues: list[str] = []

        try:
            qc: MeshQcResult = inspect(artifact.path)
        except Exception as e:  # noqa: BLE001 - unreadable file is a hard fail
            return Verdict(
                score=0.0, passed=False, issues=[f"mesh unreadable: {e}"],
                judged="artifact", judge_name=self.name,
            )

        # --- Poly budget (weighted, not gated) ---
        if qc.tri_count > TRI_BUDGET_MAX:
            budget_score = max(0.0, 1.0 - (qc.tri_count - TRI_BUDGET_MAX) / TRI_BUDGET_MAX)
            issues.append(f"over tri budget: {qc.tri_count} > {TRI_BUDGET_MAX}")
        elif qc.tri_count > TRI_BUDGET_WARN:
            budget_score = 0.85
        elif qc.tri_count == 0:
            budget_score = 0.0
            issues.append("zero triangles — empty or failed mesh")
        else:
            budget_score = 1.0

        # --- Watertightness (weighted -- some valid game assets are
        # intentionally open, e.g. a single-sided leaf or cloth plane,
        # so this is a strong signal, not an absolute requirement) ---
        watertight_score = 1.0 if qc.is_watertight else 0.6
        if not qc.is_watertight:
            issues.append("mesh is not watertight")

        # --- UV presence (weighted) ---
        uv_score = 1.0 if qc.has_uv else 0.5
        if not qc.has_uv:
            issues.append("no UV map found — texturing will fail downstream")

        # --- Degenerate faces (weighted) ---
        if qc.degenerate_face_ratio > MAX_DEGENERATE_RATIO:
            degenerate_score = max(0.0, 1.0 - qc.degenerate_face_ratio)
            issues.append(
                f"degenerate face ratio {qc.degenerate_face_ratio:.3f} exceeds {MAX_DEGENERATE_RATIO}"
            )
        else:
            degenerate_score = 1.0

        # --- Bounding box sanity (hard-fail only if the mesh is
        # essentially a point or a line -- genuinely broken, not a
        # judgment call) ---
        extent = qc.bounding_box_extent
        if max(extent) < 1e-4:
            return Verdict(
                score=0.0, passed=False,
                issues=["degenerate bounding box — mesh has no real extent"],
                judged="artifact", judge_name=self.name,
            )

        weights = {"budget": 0.30, "watertight": 0.25, "uv": 0.25, "degenerate": 0.20}
        score = (
            weights["budget"] * budget_score
            + weights["watertight"] * watertight_score
            + weights["uv"] * uv_score
            + weights["degenerate"] * degenerate_score
        )

        # Judge sets passed from ITS OWN score, per the standing rule --
        # nothing upstream gets to claim done over this.
        passed = score >= self.pass_threshold

        return Verdict(
            score=round(score, 3), passed=passed, issues=issues,
            judged="artifact", judge_name=self.name,
        )

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict:
        return self.judge(artifact, task.goal if task is not None else None)
