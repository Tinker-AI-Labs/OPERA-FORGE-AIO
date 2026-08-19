"""Spec 6 -- bible and ledger are separate stores, and stay separate."""

import json

import pytest

from opera.bible import BibleWriter, LedgerWriter, ProjectStore, estimate_tokens
from opera.config import OperaConfig
from opera.errors import ProjectStoreError
from opera.schemas import Artifact, Bible, Ledger, LedgerEntry, Run, Task, TaskStatus


def test_add_and_render():
    b, w = Bible(), BibleWriter()
    w.add(b, "characters", "Mira, a lighthouse keeper.")
    w.add(b, "style", "Cold blue palette.")
    ctx = w.context(b)
    assert "Mira" in ctx and "Cold blue" in ctx
    assert "## CHARACTERS" in ctx and "## STYLE" in ctx


def test_dedup_is_whitespace_and_case_insensitive():
    b, w = Bible(), BibleWriter()
    assert w.add(b, "facts", "The tower is red.") is not None
    assert w.add(b, "facts", "  the   TOWER is red.  ") is None
    assert len(b.entries) == 1


def test_same_text_in_different_categories_is_not_a_duplicate():
    b, w = Bible(), BibleWriter()
    w.add(b, "facts", "Red tower.")
    assert w.add(b, "style", "Red tower.") is not None
    assert len(b.entries) == 2


def test_readding_as_pinned_upgrades_existing_entry():
    b, w = Bible(), BibleWriter()
    w.add(b, "facts", "Canon fact.")
    w.add(b, "facts", "Canon fact.", pinned=True)
    assert b.entries[0].pinned is True


def test_per_category_cap_keeps_most_recent():
    b = Bible()
    w = BibleWriter(OperaConfig(context_per_category=3))
    for i in range(10):
        w.add(b, "facts", f"fact number {i}")
    ctx = w.context(b)
    assert "fact number 9" in ctx
    assert "fact number 0" not in ctx
    assert ctx.count("fact number") == 3


def test_cap_equal_to_pinned_count_does_not_leak_everything():
    """Regression: a `rest[-0:]` slice would return the entire list."""
    b = Bible()
    w = BibleWriter(OperaConfig(context_per_category=2))
    w.add(b, "facts", "pinned one", pinned=True)
    w.add(b, "facts", "pinned two", pinned=True)
    for i in range(5):
        w.add(b, "facts", f"unpinned {i}")
    ctx = w.context(b)
    assert "unpinned" not in ctx


def test_context_capped_by_token_estimate_not_item_count():
    b = Bible()
    w = BibleWriter(OperaConfig(context_per_category=50, context_token_budget=60))
    for i in range(50):
        w.add(b, "facts", f"a reasonably long established fact, number {i}, with detail")
    ctx = w.context(b)
    assert estimate_tokens(ctx) <= 120  # bounded, nowhere near 50 entries
    assert "number 49" in ctx  # newest survives trimming
    assert "number 0" not in ctx


def test_pinned_entries_survive_the_token_budget():
    b = Bible()
    w = BibleWriter(OperaConfig(context_per_category=50, context_token_budget=20))
    w.add(b, "facts", "PINNED CANON that must always be present", pinned=True)
    for i in range(30):
        w.add(b, "facts", f"filler fact {i} with some length to it")
    assert "PINNED CANON" in w.context(b)


def test_empty_bible_renders_empty():
    assert BibleWriter().context(Bible()) == ""


def test_blank_text_is_not_added():
    b, w = Bible(), BibleWriter()
    assert w.add(b, "facts", "   ") is None
    assert b.entries == []


def test_record_artifact_curates_an_excerpt_not_telemetry():
    b, w = Bible(), BibleWriter()
    task = Task(goal="Write scene one", role="writer", kind="text", status=TaskStatus.DONE, attempts=2)
    art = Artifact(task_id=task.id, kind="text", content="The lamp turned.", producer="writer")
    w.record_artifact(b, task, art)
    ctx = w.context(b)
    assert "The lamp turned." in ctx
    assert "attempts" not in ctx
    assert "status=" not in ctx


def test_artifact_excerpt_is_bounded():
    art = Artifact(content="x" * 5000)
    assert len(art.excerpt(limit=100)) <= 110


def test_artifact_without_content_falls_back_to_path():
    art = Artifact(kind="image", path="/out/frame.png")
    assert "/out/frame.png" in art.excerpt()


def test_ledger_has_no_context_method():
    """Structural guarantee that telemetry cannot be rendered into a prompt."""
    assert not hasattr(LedgerWriter, "context")


def test_ledger_records_verdict_telemetry():
    from opera.schemas import Verdict

    ledger, lw = Ledger(), LedgerWriter()
    run = Run(goal="g")
    task = Task(goal="Write", role="writer", kind="text", status=TaskStatus.DONE, attempts=2)
    task.artifacts.append(
        Artifact(producer="writer", verdict=Verdict(score=0.9, passed=True, issues=[],
                                                    judged="artifact", judge_name="llm"))
    )
    lw.record_task(ledger, run, task, duration_s=1.5, model="qwen3:8b")
    e = ledger.entries[0]
    assert e.score == 0.9 and e.judged == "artifact" and e.attempts == 2
    assert e.model == "qwen3:8b" and e.duration_s == 1.5


def test_compact_rolls_old_entries_into_a_dated_summary():
    ledger, lw = Ledger(), LedgerWriter()
    for i in range(120):
        lw.record(ledger, LedgerEntry(event="task_complete", status="done", score=0.5))
    assert lw.compact(ledger, threshold=100, keep=10) is True
    assert len(ledger.entries) == 10
    assert len(ledger.summaries) == 1
    assert "compacted 110 entries" in ledger.summaries[0]
    assert "mean_score=0.50" in ledger.summaries[0]


def test_compact_is_a_noop_below_threshold():
    ledger, lw = Ledger(), LedgerWriter()
    lw.record(ledger, LedgerEntry(event="x", status="done"))
    assert lw.compact(ledger, threshold=100) is False


def test_store_roundtrip(tmp_path):
    store = ProjectStore(tmp_path)
    p = store.create("Lighthouse", "videa")
    BibleWriter().add(p.bible, "facts", "The tower is red.")
    store.save(p)
    loaded = store.load(p.id)
    assert loaded.name == "Lighthouse"
    assert loaded.bible.entries[0].text == "The tower is red."


def test_store_write_is_atomic(tmp_path):
    """No .tmp litter, and the target is only ever replaced whole."""
    store = ProjectStore(tmp_path)
    p = store.create("A", "videa")
    store.save(p)
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads(store.path_for(p.id).read_text())["name"] == "A"


def test_store_rejects_traversal_ids(tmp_path):
    store = ProjectStore(tmp_path)
    assert store.path_for("../evil").parent == tmp_path
    with pytest.raises(ProjectStoreError):
        store.path_for("..")


def test_load_missing_project_raises(tmp_path):
    with pytest.raises(ProjectStoreError):
        ProjectStore(tmp_path).load("nope")


def test_load_corrupt_project_raises(tmp_path):
    store = ProjectStore(tmp_path)
    store.path_for("broken").write_text("{not json")
    with pytest.raises(ProjectStoreError):
        store.load("broken")


def test_list_projects_skips_unreadable_files(tmp_path):
    store = ProjectStore(tmp_path)
    store.create("Good", "videa")
    (tmp_path / "junk.json").write_text("{oops")
    assert [p["name"] for p in store.list_projects()] == ["Good"]
