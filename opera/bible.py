"""Project memory, in two stores that must not be mixed (spec 6).

The prototype fed telemetry into prompts, so models read strings like
``task t1 (writer/text) -> status=done attempts=1`` as if that were narrative
context. Here:

* **Bible**  -- what models read. Categorised, deduplicated, bounded.
* **Ledger** -- what humans and dashboards read. Never enters a prompt.

Both are JSON on disk per project. No vector DB (spec 11).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from .config import OperaConfig
from .errors import ProjectStoreError
from .schemas import (
    Artifact,
    Bible,
    BibleEntry,
    Ledger,
    LedgerEntry,
    Project,
    Run,
    Task,
)

# Categories the core understands. Engines may add their own freely -- these are
# only the ordering hints for rendering.
CATEGORY_ORDER = ("brief", "style", "characters", "facts", "decisions", "artifacts")

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def _fingerprint(category: str, text: str) -> str:
    return hashlib.sha256(f"{category}\x00{_norm(text)}".encode()).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """Cheap, deliberately conservative estimate (~4 chars/token)."""
    return max(1, (len(text) + 3) // 4)


class BibleWriter:
    """Operations on a project's Bible. Holds no per-run state of its own."""

    def __init__(self, config: OperaConfig | None = None) -> None:
        self.config = config or OperaConfig()

    def add(
        self,
        bible: Bible,
        category: str,
        text: str,
        *,
        pinned: bool = False,
        source_task: str | None = None,
    ) -> BibleEntry | None:
        """Add an entry unless an equivalent one already exists.

        Returns the new entry, or ``None`` if it was a duplicate.
        """
        text = (text or "").strip()
        if not text:
            return None
        fp = _fingerprint(category, text)
        for existing in bible.entries:
            if _fingerprint(existing.category, existing.text) == fp:
                if pinned and not existing.pinned:
                    existing.pinned = True
                return None
        entry = BibleEntry(category=category, text=text, pinned=pinned, source_task=source_task)
        bible.entries.append(entry)
        return entry

    def record_artifact(self, bible: Bible, task: Task, artifact: Artifact) -> BibleEntry | None:
        """Curate one completed artifact into the Bible.

        An excerpt, labelled by goal -- not the task's telemetry.
        """
        excerpt = artifact.excerpt()
        if not excerpt:
            return None
        return self.add(
            bible,
            "artifacts",
            f"{task.goal.strip()}:\n{excerpt}",
            source_task=task.id,
        )

    def context(
        self,
        bible: Bible,
        *,
        token_budget: int | None = None,
        per_category: int | None = None,
        extra: Sequence[str] = (),
    ) -> str:
        """Render the Bible to a prompt-ready string.

        Capped by token estimate rather than item count (spec 6), preferring
        pinned entries and then the most recent -- when a run is long, the
        newest facts are the ones the next task actually needs.
        """
        budget = token_budget if token_budget is not None else self.config.context_token_budget
        cap = per_category if per_category is not None else self.config.context_per_category

        by_cat: dict[str, list[BibleEntry]] = {}
        for entry in bible.entries:
            by_cat.setdefault(entry.category, []).append(entry)

        ordered_cats = [c for c in CATEGORY_ORDER if c in by_cat]
        ordered_cats += sorted(c for c in by_cat if c not in CATEGORY_ORDER)

        selected: list[tuple[str, BibleEntry]] = []
        for cat in ordered_cats:
            entries = by_cat[cat]
            pinned = [e for e in entries if e.pinned]
            rest = sorted((e for e in entries if not e.pinned), key=lambda e: e.created_at)
            # Pinned entries are explicit human intent and always survive the
            # per-category cap; the cap then bounds how much recent material
            # rides along with them.
            room = cap - len(pinned)
            keep = pinned + (rest[-room:] if room > 0 else [])
            keep.sort(key=lambda e: (not e.pinned, e.created_at))
            selected.extend((cat, e) for e in keep)

        lines: list[str] = []
        used = 0
        current_cat = ""
        # Walk in reverse so that, when trimming, we drop the oldest first.
        rendered: list[str] = []
        for cat, entry in reversed(selected):
            chunk = f"- {entry.text}"
            cost = estimate_tokens(chunk) + 2
            if used + cost > budget and not entry.pinned:
                continue
            used += cost
            rendered.append(f"{cat}\x00{chunk}")
        for item in reversed(rendered):
            cat, chunk = item.split("\x00", 1)
            if cat != current_cat:
                lines.append(f"\n## {cat.upper()}")
                current_cat = cat
            lines.append(chunk)

        for item in extra:
            if item.strip():
                lines.append(item.strip())
        return "\n".join(lines).strip()


class LedgerWriter:
    """Telemetry. Deliberately has no ``context()`` -- it must never be rendered
    into a prompt."""

    def __init__(self, config: OperaConfig | None = None) -> None:
        self.config = config or OperaConfig()

    def record(self, ledger: Ledger, entry: LedgerEntry) -> LedgerEntry:
        ledger.entries.append(entry)
        return entry

    def record_task(
        self,
        ledger: Ledger,
        run: Run,
        task: Task,
        *,
        event: str = "task_complete",
        duration_s: float | None = None,
        model: str | None = None,
    ) -> LedgerEntry:
        verdict = task.verdict
        artifact = task.artifact
        return self.record(
            ledger,
            LedgerEntry(
                run_id=run.id,
                task_id=task.id,
                event=event,
                status=str(task.status),
                attempts=task.attempts,
                score=verdict.score if verdict else None,
                judged=verdict.judged if verdict else None,
                judge_name=verdict.judge_name if verdict else None,
                producer=artifact.producer if artifact else None,
                model=model,
                duration_s=duration_s,
                detail={
                    "role": task.role,
                    "kind": task.kind,
                    "error": task.error,
                    "deferred_reason": task.deferred_reason,
                    "corrections": list(task.corrections),
                },
            ),
        )

    def compact(self, ledger: Ledger, *, threshold: int | None = None, keep: int = 100) -> bool:
        """Roll old entries into a dated summary once the ledger gets large."""
        limit = threshold if threshold is not None else self.config.ledger_compact_threshold
        if len(ledger.entries) <= limit:
            return False
        old, ledger.entries = ledger.entries[:-keep], ledger.entries[-keep:]
        by_status: dict[str, int] = {}
        scores = [e.score for e in old if e.score is not None]
        for e in old:
            by_status[e.status] = by_status.get(e.status, 0) + 1
        span = ""
        if old:
            span = f"{old[0].at.date().isoformat()}..{old[-1].at.date().isoformat()}"
        avg = f"{sum(scores) / len(scores):.2f}" if scores else "n/a"
        ledger.summaries.append(
            f"[{datetime.now(timezone.utc).date().isoformat()}] compacted {len(old)} entries "
            f"({span}); statuses={by_status}; mean_score={avg}"
        )
        return True


class ProjectStore:
    """JSON-on-disk project persistence.

    Writes are atomic (temp file + ``os.replace``) because the storage this runs
    on can drop mid-write, and because two concurrent runs on different projects
    must never be able to leave a half-written file behind (spec 4.7).
    """

    def __init__(self, root: str | Path | None = None, config: OperaConfig | None = None) -> None:
        self.config = config or OperaConfig()
        self.root = Path(root) if root is not None else self.config.projects_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, project_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", project_id)
        if not safe or safe in {".", ".."}:
            raise ProjectStoreError(f"unusable project id: {project_id!r}")
        return self.root / f"{safe}.json"

    def create(self, name: str, engine: str, *, project_id: str | None = None) -> Project:
        project = Project(name=name, engine=engine, **({"id": project_id} if project_id else {}))
        self.save(project)
        return project

    def exists(self, project_id: str) -> bool:
        return self.path_for(project_id).exists()

    def load(self, project_id: str) -> Project:
        path = self.path_for(project_id)
        if not path.exists():
            raise ProjectStoreError(f"no such project: {project_id}")
        try:
            return Project.model_validate_json(path.read_text())
        except Exception as exc:
            raise ProjectStoreError(f"project {project_id} is unreadable: {exc}") from exc

    def save(self, project: Project) -> Path:
        project.updated_at = datetime.now(timezone.utc)
        path = self.path_for(project.id)
        payload = project.model_dump_json(indent=2)
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(self.root), prefix=f".{path.stem}.", suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except Exception as exc:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise ProjectStoreError(f"could not save project {project.id}: {exc}") from exc
        return path

    def list_projects(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            out.append(
                {
                    "id": data.get("id", path.stem),
                    "name": data.get("name", ""),
                    "engine": data.get("engine", ""),
                    "updated_at": data.get("updated_at", ""),
                }
            )
        return out
