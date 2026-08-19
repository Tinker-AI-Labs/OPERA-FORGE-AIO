"""Spec 4.1 / 4.2 -- the per-task rules the prototype violated.

These are acceptance criteria, not suggestions.
"""

import pytest

from opera.config import LoopConfig, RoleConfig
from opera.errors import JudgeError, ProducerError
from opera.loop import execute_task
from opera.schemas import Task, TaskStatus, Verdict
from tests.doubles import (
    ApprovingGate,
    ExplodingJudge,
    GoesOfflineProducer,
    ScriptedJudge,
    ScriptedProducer,
    UnavailableProducer,
)


def _task(goal="Write scene one", role="writer", kind="text") -> Task:
    return Task(goal=goal, role=role, kind=kind)


# --- 4.1 every stored artifact carries its own verdict -----------------------

async def test_stored_artifact_and_verdict_describe_the_same_bytes():
    task = _task()
    producer = ScriptedProducer(["bad draft", "good draft"])
    judge = ScriptedJudge([False, True])

    await execute_task(task, producer, judge, "ctx", config=LoopConfig(max_attempts=2))

    assert task.artifact.content == "good draft"
    assert task.artifact.verdict.passed is True
    # The judge's last call was on the artifact that got stored, not a successor.
    judged_artifact = judge.seen[-1][1]
    assert judged_artifact is task.artifact
    assert judged_artifact.content == task.artifact.content


async def test_every_artifact_in_the_history_has_a_verdict():
    task = _task()
    await execute_task(task, ScriptedProducer(["a", "b", "c"]), ScriptedJudge([False, False, True]),
                       "ctx", config=LoopConfig(max_attempts=3))
    assert len(task.artifacts) == 3
    assert all(a.verdict is not None for a in task.artifacts)
    assert [a.attempt for a in task.artifacts] == [1, 2, 3]


async def test_final_revision_is_never_left_unjudged():
    """The prototype's bug: revise, then exit without judging the revision."""
    task = _task()
    producer = ScriptedProducer(["v1", "v2"])
    judge = ScriptedJudge([False, False])  # never passes

    await execute_task(task, producer, judge, "ctx", config=LoopConfig(max_attempts=2))

    assert producer.calls == 2
    assert judge.calls == 2  # the revision WAS judged
    assert task.artifact.content == "v2"
    assert task.artifact.verdict is not None
    assert task.artifact.verdict.passed is False


# --- 4.2 max_attempts counts total productions -------------------------------

async def test_max_attempts_two_produces_at_most_two_artifacts():
    task = _task()
    producer = ScriptedProducer(["1", "2", "3", "4", "5"])
    judge = ScriptedJudge([False, False, False, False, False])

    await execute_task(task, producer, judge, "ctx", config=LoopConfig(max_attempts=2))

    assert producer.calls == 2, "max_attempts=2 means two productions total, not two revisions"
    assert len(task.artifacts) == 2
    assert task.attempts == 2


async def test_max_attempts_one_means_no_revision_pass():
    task = _task()
    producer = ScriptedProducer(["only"])
    await execute_task(task, producer, ScriptedJudge([False]), "ctx", config=LoopConfig(max_attempts=1))
    assert producer.calls == 1


async def test_passing_first_time_spends_one_attempt():
    task = _task()
    producer = ScriptedProducer(["good"])
    await execute_task(task, producer, ScriptedJudge([True]), "ctx", config=LoopConfig(max_attempts=4))
    assert producer.calls == 1
    assert task.status is TaskStatus.DONE


async def test_role_config_overrides_loop_max_attempts():
    task = _task()
    producer = ScriptedProducer(["a", "b", "c", "d"])
    role = RoleConfig(model="m", max_attempts=3)
    await execute_task(task, producer, ScriptedJudge([False] * 5), "ctx",
                       config=LoopConfig(max_attempts=1), role_config=role)
    assert producer.calls == 3


# --- revision briefs ---------------------------------------------------------

async def test_revision_brief_carries_prior_artifact_and_issues():
    task = _task()
    judge = ScriptedJudge([
        Verdict(score=0.2, passed=False, issues=["too short", "wrong tone"],
                judged="artifact", judge_name="j"),
        Verdict(score=0.9, passed=True, issues=[], judged="artifact", judge_name="j"),
    ])
    producer = ScriptedProducer(["short", "longer and better"])

    await execute_task(task, producer, judge, "the context", config=LoopConfig(max_attempts=2))

    first, second = producer.briefs
    assert first.prior is None and first.issues == [] and first.is_revision is False
    assert second.is_revision is True
    assert second.prior.content == "short"
    assert second.issues == ["too short", "wrong tone"]
    assert second.attempt == 2


async def test_brief_never_carries_the_project():
    task = _task()
    producer = ScriptedProducer(["x"])
    await execute_task(task, producer, ScriptedJudge([True]), "ctx")
    brief = producer.briefs[0]
    assert not hasattr(brief, "project")
    assert set(brief.model_dump()) == {
        "task_id", "goal", "kind", "role", "context", "attempt", "prior", "issues", "params"
    }


async def test_context_reaches_both_producer_and_judge():
    task = _task()
    producer = ScriptedProducer(["x"])
    judge = ScriptedJudge([True])
    await execute_task(task, producer, judge, "ESTABLISHED FACTS")
    assert producer.briefs[0].context == "ESTABLISHED FACTS"
    assert judge.seen[0][2] == "ESTABLISHED FACTS"


# --- unavailable producers never fake output ---------------------------------

async def test_unavailable_producer_defers_and_is_never_called():
    task = _task(role="painter", kind="image")
    producer = UnavailableProducer()
    judge = ScriptedJudge([True])

    await execute_task(task, producer, judge, "ctx")

    assert task.status is TaskStatus.DEFERRED
    assert "unavailable" in task.deferred_reason
    assert producer.calls == 0
    assert judge.calls == 0
    assert task.artifacts == []          # no synthetic content
    assert task.verdict is None          # and no fabricated score


async def test_producer_going_offline_mid_run_defers_rather_than_failing():
    task = _task(role="scorer", kind="audio")
    await execute_task(task, GoesOfflineProducer(), ScriptedJudge([True]), "ctx")
    assert task.status is TaskStatus.DEFERRED
    assert "comfyui went away" in task.deferred_reason
    assert task.artifacts == []


async def test_producer_going_offline_after_one_pass_keeps_the_judged_work():
    from opera.errors import ProducerUnavailable

    task = _task()
    producer = ScriptedProducer(["draft"], raises=ProducerUnavailable("host died"),
                                raise_on_attempt=2)
    await execute_task(task, producer, ScriptedJudge([False, False]), "ctx",
                       config=LoopConfig(max_attempts=3))
    assert task.status is TaskStatus.DEFERRED
    assert len(task.artifacts) == 1
    assert task.artifacts[0].verdict is not None


# --- error surfaces ----------------------------------------------------------

async def test_producer_exception_propagates_for_the_runner_to_contain():
    task = _task()
    producer = ScriptedProducer(raises=RuntimeError("comfy blew up"))
    with pytest.raises(RuntimeError, match="comfy blew up"):
        await execute_task(task, producer, ScriptedJudge([True]), "ctx")


async def test_judge_exception_is_typed():
    task = _task()
    with pytest.raises(JudgeError, match="vision model crashed"):
        await execute_task(task, ScriptedProducer(["x"]), ExplodingJudge(), "ctx")


async def test_producer_returning_none_is_an_error_not_a_silent_pass():
    class NullProducer:
        name, kind, available = "null", "text", True

        async def produce(self, brief):
            return None

    with pytest.raises(ProducerError):
        await execute_task(_task(), NullProducer(), ScriptedJudge([True]), "ctx")


# --- outcome reporting -------------------------------------------------------

async def test_exhausted_review_fails_the_task_but_keeps_the_artifact():
    task = _task()
    await execute_task(task, ScriptedProducer(["a", "b"]), ScriptedJudge([0.3, 0.4]),
                       "ctx", config=LoopConfig(max_attempts=2))
    assert task.status is TaskStatus.FAILED
    assert task.error.startswith("review_not_passed:")   # distinguishable from a crash
    assert task.artifact.content == "b"                   # work is not thrown away


async def test_outcome_reports_productions_not_revisions():
    outcome = await execute_task(_task(), ScriptedProducer(["a", "b"]), ScriptedJudge([False, True]),
                                 "ctx", config=LoopConfig(max_attempts=2))
    assert outcome.productions == 2
    assert outcome.passed is True


async def test_judged_field_propagates_to_the_stored_verdict():
    task = _task()
    judge = ScriptedJudge([True], judged="plan", name="musica-composite")
    await execute_task(task, ScriptedProducer(["x"]), judge, "ctx")
    assert task.artifact.verdict.judged == "plan"
    assert task.artifact.verdict.judge_name == "musica-composite"


async def test_artifact_metadata_is_stamped_by_the_loop():
    task = _task()
    await execute_task(task, ScriptedProducer(["x"], name="writer", kind="text"),
                       ScriptedJudge([True]), "ctx")
    art = task.artifact
    assert art.task_id == task.id and art.producer == "writer" and art.kind == "text"


# --- human gate --------------------------------------------------------------

async def test_gate_refusal_parks_the_task_awaiting_review():
    task = _task()
    gate = ApprovingGate(approve=False)
    await execute_task(task, ScriptedProducer(["x"]), ScriptedJudge([True]), "ctx", gate=gate)
    assert task.status is TaskStatus.AWAITING_REVIEW
    assert gate.calls == 1


async def test_gate_approval_leaves_the_task_done():
    task = _task()
    gate = ApprovingGate(approve=True)
    await execute_task(task, ScriptedProducer(["x"]), ScriptedJudge([True]), "ctx", gate=gate)
    assert task.status is TaskStatus.DONE


async def test_gate_is_not_consulted_for_a_failed_task():
    task = _task()
    gate = ApprovingGate(approve=True)
    await execute_task(task, ScriptedProducer(["x"]), ScriptedJudge([False]), "ctx",
                       config=LoopConfig(max_attempts=1), gate=gate)
    assert gate.calls == 0


async def test_revise_only_with_issues_stops_a_blind_reroll():
    task = _task()
    producer = ScriptedProducer(["a", "b"])
    judge = ScriptedJudge([Verdict(score=0.4, passed=False, issues=[], judged="artifact",
                                   judge_name="j")])
    await execute_task(task, producer, judge, "ctx",
                       config=LoopConfig(max_attempts=3, revise_only_with_issues=True))
    assert producer.calls == 1


# --- best_of_n policy (2026-08-19) --------------------------------------------

async def test_best_of_n_keeps_the_highest_scoring_artifact():
    """A ranking judge that never clears an absolute pass_threshold is still
    useful under best_of_n -- it just isn't a gate."""
    task = _task(role="retoucher", kind="image")
    producer = ScriptedProducer(["draft-1", "draft-2", "draft-3"])
    judge = ScriptedJudge([0.4, 0.9, 0.6])

    await execute_task(task, producer, judge, "ctx",
                       config=LoopConfig(max_attempts=3, policy="best_of_n"))

    assert task.artifact.content == "draft-2"
    assert task.artifact.verdict.score == pytest.approx(0.9)
    assert task.status is TaskStatus.DONE


async def test_best_of_n_retains_every_attempt_not_just_the_winner():
    """Losers are not discarded -- same never-discard shape rerun_task uses."""
    task = _task(role="retoucher", kind="image")
    producer = ScriptedProducer(["draft-1", "draft-2", "draft-3"])
    judge = ScriptedJudge([0.4, 0.9, 0.6])

    await execute_task(task, producer, judge, "ctx",
                       config=LoopConfig(max_attempts=3, policy="best_of_n"))

    assert task.attempts == 3
    assert [a.content for a in task.artifacts] == ["draft-1", "draft-2", "draft-3"]
    assert [a.verdict.score for a in task.artifacts] == pytest.approx([0.4, 0.9, 0.6])
    # The winner is marked chosen; task.artifact resolves to it even though
    # it is not the last one produced.
    assert task.chosen_artifact_id == task.artifacts[1].id
    assert task.artifact is task.artifacts[1]


async def test_best_of_n_verdict_is_unmodified_by_the_selection_policy():
    """best_of_n changes what the loop does with a verdict, not what the
    judge reported on it."""
    task = _task(role="retoucher", kind="image")
    producer = ScriptedProducer(["draft-1", "draft-2"])
    judge = ScriptedJudge([
        Verdict(score=0.3, passed=False, issues=["too dark"], judged="artifact", judge_name="j"),
        Verdict(score=0.9, passed=True, issues=[], judged="artifact", judge_name="j"),
    ])

    await execute_task(task, producer, judge, "ctx",
                       config=LoopConfig(max_attempts=2, policy="best_of_n"))

    loser = task.artifacts[0]
    assert loser.verdict.score == 0.3
    assert loser.verdict.issues == ["too dark"]
    assert task.artifact.verdict.issues == []


async def test_best_of_n_stops_early_once_an_attempt_clears_the_ceiling():
    """Must not silently mean 'always spend N renders' -- a configured
    ceiling lets a clearly good early attempt end the batch."""
    task = _task(role="retoucher", kind="image")
    producer = ScriptedProducer(["draft-1", "draft-2", "draft-3", "draft-4"])
    judge = ScriptedJudge([0.5, 0.95, 0.6, 0.6])

    await execute_task(task, producer, judge, "ctx",
                       config=LoopConfig(max_attempts=4, policy="best_of_n", ceiling=0.9))

    assert producer.calls == 2, "should stop right after the 0.95 attempt clears the 0.9 ceiling"
    assert task.attempts == 2
    assert task.artifact.content == "draft-2"


async def test_best_of_n_without_a_ceiling_spends_the_full_budget():
    task = _task(role="retoucher", kind="image")
    producer = ScriptedProducer(["draft-1", "draft-2", "draft-3"])
    judge = ScriptedJudge([0.99, 0.99, 0.99])  # every attempt would "clear" any reasonable bar

    await execute_task(task, producer, judge, "ctx",
                       config=LoopConfig(max_attempts=3, policy="best_of_n"))

    assert producer.calls == 3, "no ceiling configured means the full N is always spent"


async def test_best_of_n_n_comes_from_config():
    task = _task(role="retoucher", kind="image")
    producer = ScriptedProducer(["a", "b", "c", "d", "e"])
    judge = ScriptedJudge([0.5, 0.5, 0.5, 0.5, 0.5])

    await execute_task(task, producer, judge, "ctx",
                       config=LoopConfig(max_attempts=5, policy="best_of_n"))

    assert producer.calls == 5


async def test_best_of_n_role_config_overrides_loop_config():
    """Same precedence as max_attempts/pass_threshold: a role_config's policy
    wins over the engine-level default."""
    task = _task(role="retoucher", kind="image")
    producer = ScriptedProducer(["draft-1", "draft-2"])
    judge = ScriptedJudge([0.3, 0.8])
    role = RoleConfig(model="m", max_attempts=2, policy="best_of_n")

    # LoopConfig defaults to threshold; role_config overrides it.
    await execute_task(task, producer, judge, "ctx",
                       config=LoopConfig(max_attempts=2, policy="threshold"),
                       role_config=role)

    assert task.artifact.content == "draft-2"
    assert task.attempts == 2, "threshold would have stopped at attempt 1 if it scored under 0.7"


async def test_default_policy_is_threshold_videa_style():
    """No policy configured anywhere -- VIDEA's text roles -- behaves exactly
    as before this feature existed: revise until a verdict passes, stopping
    immediately rather than spending the rest of the attempt budget."""
    task = _task()
    producer = ScriptedProducer(["good-enough", "would-never-run"])
    judge = ScriptedJudge([True])

    await execute_task(task, producer, judge, "ctx", config=LoopConfig(max_attempts=2))

    assert producer.calls == 1, "threshold stops as soon as a verdict passes"
    assert task.artifact.content == "good-enough"


async def test_best_of_n_unavailable_producer_mid_batch_defers_without_losing_attempts():
    task = _task(role="retoucher", kind="image")
    producer = GoesOfflineProducer(name="flaky", kind="image")

    await execute_task(task, producer, ScriptedJudge([0.5]), "ctx",
                       config=LoopConfig(max_attempts=3, policy="best_of_n"))

    assert task.status is TaskStatus.DEFERRED
    assert task.artifacts == []
