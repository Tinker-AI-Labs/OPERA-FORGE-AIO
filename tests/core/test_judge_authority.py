"""Defect-1 class: model output must never be treated as authority.

The original bug was `LLMJudge`/`VisionJudge` trusting a model's own `passed`
claim over its own `score`. Three siblings of that same mistake, per the
2026-08-18 audit:

  (a) `Verdict.judged` must be hardcoded per judge implementation, never read
      from model output.
  (b) `CompositeJudge` must not take a subordinate judge's `.passed` at face
      value -- a subordinate is itself just a claim until reconciled against
      its own score.
  (c) A score must be clamped/coerced at the parse boundary, and anything
      genuinely unparseable (missing, null, non-numeric, non-finite) is a
      judge failure, not a silent 0.0 that might coincidentally clear a low
      threshold.

(b) and (c) each fixed a real, demonstrable defect in judges.py -- the tests
below failed against the code as it stood before this pass. (a) was audited
and found already correct throughout the codebase (every `Verdict(...)`
construction site uses a hardcoded or judge-derived `judged`, never
`data.get("judged")`); its test is a standing invariant guard, not a
before/after fix -- see the audit note on the test itself.
"""

from __future__ import annotations

import json
import math

import pytest

from opera.errors import JudgeError
from opera.judges import CompositeJudge, DeterministicJudge, LLMJudge, VisionJudge
from opera.llm.stub import StubLLMClient
from opera.schemas import Artifact, Task, Verdict

TASK = Task(goal="Write it", role="writer", kind="text")


def _stub(payload):
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return StubLLMClient(default=body)


# ── (a) judged is never model-sourced -----------------------------------

class _NaiveJudge:
    """A subordinate that does NOT reconcile -- the shape a careless
    engine-specific or third-party Judge could realistically take."""

    def __init__(self, score, passed, judged="artifact", name="naive"):
        self.name = name
        self._v = Verdict(score=score, passed=passed, issues=[], judged=judged,
                          judge_name=name)

    async def evaluate(self, task, artifact, context):
        return self._v


async def test_judged_is_never_read_from_model_output():
    """Audit invariant, not a bug fix: every judge that talks to a model only
    ever sets `judged` from a hardcoded default or constructor argument, even
    when the model's JSON smuggles a `judged` key of its own."""
    llm_client = _stub({"score": 0.9, "passed": True, "issues": [],
                        "judged": "MODEL-INJECTED-VALUE"})
    vision_client = _stub({
        "subject_present": True,
        "elements": [], "artifacts_present": False, "composition_intact": True,
        "judged": "MODEL-INJECTED-VALUE",
    })

    llm_verdict = await LLMJudge(llm_client, model="m", judged="artifact").evaluate(
        TASK, Artifact(content="t"), "")
    vision_verdict = await VisionJudge(vision_client, model="v", judged="artifact").evaluate(
        TASK, Artifact(kind="image", meta={"image_b64": "QUJD"}), "")

    assert llm_verdict.judged == "artifact"
    assert vision_verdict.judged == "artifact"
    assert "MODEL-INJECTED-VALUE" not in (llm_verdict.judged, vision_verdict.judged)

    composite = CompositeJudge([_NaiveJudge(0.9, True, judged="artifact")])
    composite_verdict = await composite.evaluate(TASK, Artifact(), "")
    assert composite_verdict.judged == "artifact"


# ── (b) CompositeJudge reconciles subordinates, not just its own model ---

async def test_composite_does_not_trust_a_low_scoring_subordinates_passed_claim():
    """The exact shape of the original bug, one level up: a subordinate judge
    that (like a naive third-party Judge, or the pre-fix LLMJudge itself
    would have) declares passed=True on a score nowhere near threshold."""
    naive = _NaiveJudge(score=0.1, passed=True, name="naive")
    composite = CompositeJudge([naive], policy="all_must_pass", threshold=0.7)

    verdict = await composite.evaluate(TASK, Artifact(), "")

    assert verdict.passed is False, (
        "a subordinate claiming passed=True on score=0.1 must not make the "
        "composite pass just because CompositeJudge trusted the claim"
    )
    assert "judge_disagreement" in verdict.detail["members"]["naive"]


async def test_composite_any_may_pass_also_reconciles_subordinates():
    naive_good_claim_bad_score = _NaiveJudge(score=0.05, passed=True, name="a")
    composite = CompositeJudge([naive_good_claim_bad_score], policy="any_may_pass",
                               threshold=0.7)
    verdict = await composite.evaluate(TASK, Artifact(), "")
    assert verdict.passed is False


async def test_composite_reconciliation_uses_each_subordinates_own_threshold():
    """A subordinate configured with a looser threshold than the composite's
    default should be judged against ITS OWN bar, not silently overridden."""
    class LooseThresholdJudge:
        name = "loose"
        threshold = 0.2

        async def evaluate(self, task, artifact, context):
            return Verdict(score=0.25, passed=True, issues=[], judged="artifact",
                           judge_name=self.name)

    composite = CompositeJudge([LooseThresholdJudge()], policy="all_must_pass",
                               threshold=0.7)
    verdict = await composite.evaluate(TASK, Artifact(), "")
    assert verdict.passed is True, "0.25 clears the subordinate's own 0.2 bar"


async def test_composite_does_not_silently_flip_an_honest_subordinate():
    """Reconciliation must not change a result that was already consistent."""
    honest = _NaiveJudge(score=0.9, passed=True, name="honest")
    composite = CompositeJudge([honest], policy="all_must_pass")
    verdict = await composite.evaluate(TASK, Artifact(), "")
    assert verdict.passed is True
    assert "judge_disagreement" not in verdict.detail["members"]["honest"]


async def test_deterministic_judge_subordinate_is_unaffected_by_reconciliation():
    """DeterministicJudge already agrees with itself (passed == all-checks-ok,
    which is consistent with score >= threshold by construction here) -- make
    sure reconciling composite subordinates doesn't second-guess a judge that
    was never wrong."""
    checks = DeterministicJudge([("ok", lambda a: (True, ""))])
    composite = CompositeJudge([checks], threshold=0.7)
    verdict = await composite.evaluate(TASK, Artifact(), "")
    assert verdict.passed is True


# ── (c) scores are clamped/coerced at the parse boundary, or it's a fail --

@pytest.mark.parametrize("bad_payload,label", [
    ({"passed": True, "issues": []}, "missing"),
    ({"score": None, "passed": True, "issues": []}, "null"),
    ({"score": "high", "passed": True, "issues": []}, "wrong-type-string"),
    ({"score": True, "passed": True, "issues": []}, "wrong-type-bool"),
    ('{"score": NaN, "passed": true, "issues": []}', "nan"),
    ('{"score": Infinity, "passed": true, "issues": []}', "infinity"),
])
async def test_llm_judge_treats_unparseable_score_as_a_failure_not_a_zero(bad_payload, label):
    client = _stub(bad_payload)
    with pytest.raises(JudgeError, match="score"):
        await LLMJudge(client, model="m", threshold=0.0).evaluate(
            TASK, Artifact(content="t"), "")


@pytest.mark.parametrize("bad_payload,label", [
    ({"elements": [], "artifacts_present": False, "composition_intact": True}, "missing-subject"),
    ({"subject_present": True, "artifacts_present": False, "composition_intact": True}, "missing-elements"),
    ({"subject_present": True, "elements": [], "composition_intact": True}, "missing-artifacts"),
    ({"subject_present": True, "elements": [], "artifacts_present": False}, "missing-composition"),
    ({"subject_present": "yes", "elements": [], "artifacts_present": False,
      "composition_intact": True}, "wrong-type-string"),
])
async def test_vision_judge_treats_unparseable_rubric_as_a_failure_not_a_default(bad_payload, label):
    """2026-08-19 rubric rewrite: VisionJudge no longer asks for or reads a
    self-reported `score` at all (llava:7b's number carried no signal --
    see project notes). The same 'unparseable is a fail, not a silent
    default' discipline from (c) now applies to the rubric fields instead."""
    client = _stub(bad_payload)
    art = Artifact(kind="image", meta={"image_b64": "QUJD"})
    with pytest.raises(JudgeError):
        await VisionJudge(client, model="v", threshold=0.0).evaluate(TASK, art, "")


async def test_missing_score_does_not_coincidentally_pass_a_low_threshold():
    """The concrete failure mode named in the task: a permissive engine sets
    threshold=0.0, and a judge that silently defaulted a missing score to 0.0
    would have this artifact PASS with a fabricated 0.0 score. It must raise
    instead."""
    client = _stub({"passed": True, "issues": []})  # no "score" key at all
    with pytest.raises(JudgeError):
        await LLMJudge(client, model="m", threshold=0.0).evaluate(
            TASK, Artifact(content="t"), "")


@pytest.mark.parametrize("value,expected", [(1.5, 1.0), (-0.2, 0.0), (2, 1.0)])
async def test_out_of_range_numeric_scores_are_clamped_not_rejected(value, expected):
    """Distinguishing case: a real (if invalid) number is coerced, not failed --
    only genuinely unparseable input is a failure."""
    client = _stub({"score": value, "passed": True, "issues": []})
    verdict = await LLMJudge(client, model="m", threshold=0.0).evaluate(
        TASK, Artifact(content="t"), "")
    assert verdict.score == expected


async def test_integer_score_is_accepted_and_coerced_to_float():
    client = _stub({"score": 1, "passed": True, "issues": []})
    verdict = await LLMJudge(client, model="m").evaluate(TASK, Artifact(content="t"), "")
    assert verdict.score == 1.0
