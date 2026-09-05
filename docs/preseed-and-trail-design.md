# Preseed (SeedSheet) & Trail Drift Tracking — Design Notes

Status: **design only, not built.** Captures a design conversation; no schema,
pipeline, or breadcrumb changes have been made yet. Written against the
current core (`opera/schemas.py`, `opera/pipeline.py`, `opera/judges.py`) so
it can be picked up as a build spec later.

## Problem

Every engine (VIDEA, GAMEA, PUBLISHA, ARTISTA) currently starts a new project
cold: no reference set to conform to, no baseline traits to generate toward,
no ground truth for the judge to score against beyond a text goal. This
mirrors the Anijam "character sheet" problem — keeping output recognizable
across different generators (or different artists, in the Anijam case)
without rigid, single-style enforcement.

## SeedSheet

A new engine-agnostic schema, `SeedSheet`, sitting alongside `Artifact` /
`Verdict` / `Task` in `opera/schemas.py`. One SeedSheet per project (or per
character/asset family if a project needs more than one), stored in the
project's story bible (`opera/memory.py`).

Consumed differently per engine, same object:

- **VIDEA** — reference images used as IP-Adapter/InstantID conditioning;
  turnaround set doubles as multi-view input.
- **GAMEA** — turnaround views become literal multi-view conditioning for
  Hunyuan3D-2 / TripoSR mesh generation; palette/material fields feed the
  texture pass.
- **PUBLISHA** — text fields only (no reference images); feeds generation
  prompts and, more importantly, a judge rubric (see "Hard don'ts" below).
- **ARTISTA / judges.py** — any SeedSheet becomes a scoring rubric: an
  artifact is checked against its locked traits, not just a text goal.

### Fields (draft)

Common core:
- `project_id`
- `title` / character or asset name
- `locked_traits: list[str]` — the non-negotiable markers (silhouette, props,
  signature elements)
- `palette: list[str]` — hex swatches
- `reference_images: list[str]` — paths, optional (VIDEA/GAMEA use these,
  PUBLISHA typically won't)

### Per-engine preseed forms

Short, filled once per new project, not per task. Existing projects skip
this and load the stored SeedSheet straight from the story bible.

**VIDEA**
1. Character name
2. Silhouette / body-shape description
3. Palette (hex swatches)
4. Signature props / clothing
5. Turnaround images if available (front / 3-4 / side / back)
6. Expression range needed

**GAMEA**
1. Asset type (character / prop / environment)
2. Target poly budget
3. Texture resolution
4. Material list
5. Rigging needed (y/n)

**PUBLISHA**
1. Title / working title
2. POV & voice (first-person journal, third-limited, script dialogue, etc.)
3. Protagonist + 2-3 defining traits
4. Domain facts the story must use correctly
5. Format target (chapter length, comic script, blog post, journal entry)
6. Hard don'ts — pre-filled defaults from observed failure modes
   (llama3.1:8b vs gemma2:9b testing, 2026): no bracketed placeholder text
   (`[Your Name]`), no character stating the story's moral aloud as dialogue,
   no narrator breaking voice.

**Note on reference images (VIDEA/GAMEA):** keep sheets at the genre/style
level, not a redraw of specific copyrighted IP — matters if a sheet is built
around licensed source material (see the retro-post-apocalyptic example
worked through earlier) and the output feeds anything commercial or
patron-facing.

### Why this also limits prompt-injection / misalignment risk

Field values become structured project data consumed by the pipeline, not
raw chat text passed straight to a model. The "hard don'ts" are enforced
mechanically by the judge (see below) on the *output artifact*, not by
trusting whatever instruction happened to come through in a chat turn — so a
manipulated or off-rails conversation can't talk the pipeline into ignoring
the constraints. This holds regardless of which model is running under
PUBLISHA (gemma2 today, potentially something else later).

## Judge rubric additions

New rubric fields in `opera/judges.py`, following the existing pattern
(`has_visual_corruption` on ARTISTA's VisionJudge):

- PUBLISHA: `has_unfilled_placeholder` (regex check for bracketed
  placeholder patterns), `has_stated_moral` (semantic check — theme spoken
  directly as dialogue rather than shown).
- VIDEA/GAMEA: conformance-to-SeedSheet check — does the artifact preserve
  `locked_traits` and `palette` from the project's sheet.

## Workflow

1. **Pick project.**
   - Existing project → skip preseed, load stored SeedSheet from the story
     bible, submit straight to `router.py`.
   - New project → step 2.
2. **New project → pick engine → fill that engine's preseed form.**
   Answers populate the SeedSheet fields directly (no freeform authoring
   step).
3. **Submit for work.** `router.py` routes to the target engine with the
   SeedSheet attached as conditioning/rubric input instead of a bare prompt.
   Produce → judge → revise runs as normal, judged against the sheet.
4. **Sheet persists** in the project's story bible — every later task in
   that project is the "existing project" fast path from step 1.

## Long-term drift tracking (trail)

Reuses the existing `breadcrumb.py` (`BreadcrumbEngine`) system rather than
inventing a new format — append-only `.trail` files, crumb types
NOTE/WATCH/RESOLVED/BLOCKER/IDEA. `breadcrumb.py` currently lives in the
ENGRAM/brain package, not in this repo; using it here means either vendoring
it into `opera/` or importing it as a shared dependency — not yet decided.

**Per-engine trails: single file, engine field on the crumb — not separate
`.trailg` / `.trailv` / `.trailz` extensions.** Rationale: anything already
pattern-matching `*.trail` (backup/dedup tooling on the float, any future
trail reader) keeps working unmodified; per-engine filtering comes from
querying the `engine` field (`gamea` / `videa` / `publisha` / `artista`)
rather than the filename. One `.trail` per project.

Drift bands reuse the threshold model already calibrated in
`engram_provenance.py` / `engram_coherence.py` (DRIFTING at 0.18, UNMOORED at
0.08 for content-grounding) rather than inventing new numbers:

- **NOTE** — judge pass, conforms to the SeedSheet.
- **WATCH** — passes the general quality bar but trending away from the
  sheet (DRIFTING-band score).
- **BLOCKER** — judge rejects specifically on sheet-conformance
  (UNMOORED-band score), not general quality.
- **RESOLVED** — a later revision brings it back in line, closes the open
  WATCH/BLOCKER.

### Prerequisite: fix B1 before relying on this

`breadcrumb.py`'s `drop()` has no file lock (tracked issue B1). This design
increases write frequency onto a project's trail — up to four engines
writing crumbs against the same file — which makes that race more likely,
not less. Fix B1 first.

## Build order

1. Fix `breadcrumb.py` B1 (file lock on `drop()`).
2. Add `engine` field to the crumb schema.
3. Wire `pipeline.py` to drop a crumb after every judge pass.
4. Add `SeedSheet` to `opera/schemas.py` + story-bible persistence in
   `opera/memory.py`.
5. Build the three preseed forms (VIDEA / GAMEA / PUBLISHA) and the
   new/existing-project lookup in `pipeline.py`.
6. Add the judge rubric fields above to `opera/judges.py`.

None of this has been implemented yet — this file is the spec to build
against.