# OPERA — Shared Orchestration Core

Build spec for a ground-up implementation. Hand this to Claude Code as the brief.

**Name:** `OPERA` (Latin *opera*, "works" — plural of *opus*; also the collaborative
art form). Fits the FABRICA / MUSICA / ARTISTA / VIDEA naming. Swap it if you want
something else — it appears only as the package name and in imports.

---

## 1. What this is

A local, engine-agnostic **produce → judge → revise → persist** loop with project
memory. Three FORGE engines are built on top of it:

| Engine | Produces | Judge |
|---|---|---|
| ARTISTA | images (FLUX.2 / ComfyUI) | local vision model |
| VIDEA | video | vision model over sampled frames |
| MUSICA | audio (ACE-Step / FluidSynth) | deterministic checks + plan review + human gate |

**FABRICA is explicitly out of scope.** Fabrication output is objectively
verifiable (compiles / manifold / fits the bounding box), not opinion-scored. It
stays a separate codebase. The only concession to a future change of mind: the
loop must call a `Judge` protocol, never a concrete reviewer class, so a
deterministic verifier could be dropped in later without touching the loop.

Everything runs locally. Ollama for LLM roles, local generators for media. No
cloud APIs, no vector DB, no web UI.

---

## 2. Repository layout

```
opera/                      # the shared core — knows nothing about any engine
  __init__.py
  schemas.py                # Task, Run, Artifact, Verdict, Project
  protocols.py               # Producer, Judge, LLMClient
  llm/
    ollama.py               # async httpx client
    stub.py                 # deterministic in-process client for tests
  planner.py                # prompt -> task list
  router.py                 # goal -> (role, kind), engine-supplied vocabulary
  loop.py                   # produce -> judge -> revise, per task
  runner.py                 # orchestrates a full run over a task list
  bible.py                  # project memory
  registry.py               # EngineSpec registration
  errors.py

engines/
  videa/
  artista/
  musica/
    spec.py                 # EngineSpec: roles, producers, judge, kinds
    producers.py
    judge.py

service/                    # service surfaces
  api.py                    # FastAPI
  cli.py

tests/
  core/
  engines/
```

Do not put the core under any engine directory. VIDEA is a consumer of OPERA, not
its host.

---

## 3. Core abstractions

### 3.1 `Producer`

The single interface for anything that makes an artifact. An LLM agent and a
ComfyUI image generator are both Producers. This is the central change from the
prototype, where LLM agents and media generators were separate code paths and
only the LLM path had a review loop.

```python
class Producer(Protocol):
    name: str
    kind: str                     # "text" | "code" | "image" | "video" | "audio"
    available: bool               # False -> task is deferred, not faked

    async def produce(self, brief: Brief) -> Artifact: ...
```

`Brief` carries the goal, the project context string, and — on a revision pass —
the prior artifact plus the judge's issues. Producers must not be given the whole
`Project`; keep the surface narrow.

Unavailable producers cause the task to be marked `deferred` with a reason. They
must never return placeholder or synthetic output. This behaviour was correct in
the prototype and must be preserved.

### 3.2 `Judge`

```python
class Verdict(BaseModel):
    score: float                  # 0.0 - 1.0
    passed: bool
    issues: list[str]
    judged: str                   # what was actually assessed: "artifact" | "plan" | "frames"
    judge_name: str

class Judge(Protocol):
    name: str
    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict: ...
```

The `judged` field is required, not decorative. MUSICA cannot assess a rendered
waveform's musicality and must report `judged="plan"` so a downstream consumer
knows the score does not cover the audio itself. VIDEA reports `judged="frames"`.
Do not let an engine emit a score that implies coverage it doesn't have.

Ship these judge implementations:

- `LLMJudge` — text/code. Scores the artifact against goal + bible context.
- `VisionJudge` — images. Multimodal model via Ollama, artifact passed as base64.
- `FrameSampleJudge` — video. Extracts N frames via ffmpeg, wraps `VisionJudge`,
  aggregates. Scores prompt adherence and cross-frame continuity only.
- `DeterministicJudge` — takes a list of check callables returning
  `(bool, message)`. MUSICA uses it for duration vs. spec, detected key/tempo vs.
  spec, clipping, leading/trailing silence.
- `CompositeJudge` — runs several judges, combines with a configurable policy
  (`all_must_pass` default). MUSICA = DeterministicJudge + LLMJudge on the plan.

### 3.3 `EngineSpec`

An engine is data, not a fork of the core:

```python
@dataclass
class EngineSpec:
    name: str
    roles: dict[str, RoleConfig]        # engine's own vocabulary
    producers: dict[str, Producer]      # keyed by role name
    judge: Judge
    kinds: frozenset[str]
    planner_system_prompt: str
    router_keywords: dict[str, list[str]]
```

Role vocabularies are per-engine and come from config. `writer / reasoner / coder`
is VIDEA's text vocabulary, not a global one. MUSICA wants
`composer / arranger / mixer`; ARTISTA wants `concept / prompt_smith / retoucher`.
There must be no module-level `AGENT_REGISTRY` and no hardcoded `TaskKind`
`Literal` in the core — kinds are validated against `spec.kinds` at runtime.

---

## 4. Loop requirements

These are the correctness rules. The prototype violated the first three; treat
them as acceptance criteria, not suggestions.

**4.1 Every stored artifact carries its own verdict.** Structure the loop as
produce → judge → (revise → judge)*. Never exit having produced a revision that
was not then judged. `artifact.verdict` and `artifact` must describe the same
bytes.

**4.2 `max_attempts` counts total productions.** `max_attempts: 2` means at most
two artifacts generated, i.e. one initial plus one revision. Assert this in a test.

**4.3 Context is recomputed per task, not per run.** Snapshotting the bible before
the task loop means task 2 has never seen task 1's output — which defeats the
entire point of project memory within a multi-scene run. Recompute
`bible.context(project)` at the top of each task iteration.

**4.4 A failing task fails that task, not the run.** Wrap each task's execution.
On any exception: mark the task `failed`, record the error on the task, continue
to the next task. A planner hallucinating `role="cinematographer"` must not
discard four completed scenes.

**4.5 Planner output is validated, never trusted.** Any role not in
`spec.producers` falls back to `router.route(goal)`. Any kind not in `spec.kinds`
falls back to the engine default. Log the correction on the task. Note that the
prototype's tests could never catch this because the stub planner only emitted
valid roles — write a test with a planner that emits garbage.

**4.6 Persist incrementally.** Save the project after each task completes, not
once at the end of the run. Runs here can span minutes per task; a crash on task
five must not lose tasks one through four.

**4.7 No shared mutable run state.** `StoryBible` and any per-run state are
locals inside `run()`, passed down explicitly — never assigned to `self` on a
long-lived object. Two concurrent FastAPI requests for different projects must
not be able to cross-write each other's files. Include a concurrency test that
runs two projects simultaneously and asserts both files are intact.

---

## 5. LLM client requirements

**5.1 Structured output is requested, not hoped for.** Pass Ollama's
`"format": "json"` for planner, judge, and router calls. Do not rely on
scraping braces out of prose.

**5.2 Handle reasoning models.** qwen3 emits `<think>` blocks by default. Strip
them before parsing, and append `/no_think` to system prompts for roles where
reasoning traces are unwanted. Without this the JSON extractor will span the
think block and fail on every planner and judge call — which presents as poor
model quality but is actually a parsing bug.

**5.3 JSON extraction is defensive.** Strip code fences, strip think blocks,
then attempt a full parse before falling back to a brace-span scan. On failure,
return a structured parse error, not `None`.

**5.4 No silent stub fallback.** The prototype defaulted to `StubOllamaClient`
when no client was passed, which turns a wiring mistake into plausible fake
output. Require the client explicitly; raise if absent. Stub selection is an
explicit caller decision (`--stub`).

**5.5 Timeouts and retries.** Per-role timeout from config. Transport errors get
bounded retries with backoff, distinct from review-driven revisions — a network
retry must not consume a revision attempt.

---

## 6. Project memory

Two separate stores. The prototype mixed them, so every prompt carried strings
like `task t1 (writer/text) -> status=done attempts=1` as narrative context.

- **Bible** — what the models read. Characters, style decisions, established
  facts, curated artifact excerpts. Categorised, deduplicated, bounded per
  category when rendered to a context string.
- **Ledger** — telemetry. Run history, task statuses, attempt counts, verdicts,
  timings, model names. Never injected into a prompt.

Both are JSON on disk per project, no external DB. Add a `compact()` that rolls
old ledger entries into a dated summary once a project exceeds a size threshold.

Cap the rendered context string by token estimate, not item count, and prefer
recent + explicitly pinned entries when trimming.

---

## 7. Routing

Keyword routing is engine-supplied. Two fixes over the prototype:

- **Media precedence is wrong.** It checked media keywords before role keywords,
  so "write a script for a video clip" routed entirely to media and produced no
  writing. A prompt can imply both; the planner should be able to emit a text
  task and a media task from one goal.
- **Keywords were too broad.** Bare `art`, `score`, `music` match "state of the
  art" and "a musical score plays in the background". Require multi-word phrases
  or explicit generation verbs for media classification.

Keep it rule-based and deterministic by default; the LLM router is an opt-in
fallback for ambiguous goals only.

---

## 8. Service surfaces

**CLI:** `run`, `health`, `serve`, `projects`, `show <run-id>`. `--engine` selects
the EngineSpec. `--stub` uses stub client and stub producers.

**FastAPI:** health, project CRUD, run submission, run status, bible read.
Runs are long — submission returns a run id immediately and status is polled.
Do not hold an HTTP connection open for a multi-minute render. Use lifespan
handlers, not the deprecated `@app.on_event`.

---

## 9. Testing

Everything runs without Ollama, without ComfyUI, without ffmpeg. Stub client,
stub producers, scripted judges.

Required cases, beyond the obvious happy paths — these are the gaps the prototype
suite left open:

- Artifact/verdict pairing: final stored artifact was judged, not a successor.
- Attempt accounting: `max_attempts=2` produces exactly two artifacts maximum.
- Planner emitting an unknown role → falls back to router, run completes.
- Producer raising mid-run → that task fails, remaining tasks still run, project
  persisted with partial results.
- Intra-run continuity: task 2's brief contains task 1's excerpt.
- Concurrency: two projects run simultaneously, neither file is corrupted.
- Unavailable producer → `deferred`, and the result contains no synthetic content.
- Mixed run (one done, one deferred) → aggregate status reports the deferral
  rather than collapsing to `done`.
- Judge `judged` field propagates to the stored verdict.
- `<think>`-wrapped JSON parses correctly.

---

## 10. Build order

1. `schemas.py`, `protocols.py`, `errors.py` — the contracts, nothing else.
2. `llm/` client + stub, with the JSON/think handling and its tests.
3. `bible.py` + ledger separation.
4. `loop.py` — with all seven rules in §4 and their tests. This is the piece
   everything else depends on being right; do not move past it until green.
5. `planner.py`, `router.py`, `runner.py`, `registry.py`.
6. VIDEA EngineSpec — port the existing text roles onto the new core. This is
   the migration checkpoint.
7. ARTISTA EngineSpec — first real non-LLM producer, first `VisionJudge`. This
   proves the Producer abstraction actually holds.
8. MUSICA EngineSpec — `CompositeJudge`, deterministic checks, human gate.
9. Service surfaces.

Steps 1–6 are the real work. If 7 requires changing the core, the abstraction is
wrong and it's cheaper to find that out at step 7 than at step 8.

> **2026-08-18 note:** §10.6 originally closed with "behaviour parity with the
> prototype, minus the bugs." That language was struck during the build: no
> prototype exists anywhere on this float to port from or check parity
> against (confirmed by search). VIDEA was built from this spec's own role
> vocabulary and the loop requirements in §4 instead — there is nothing to
> prove parity with here.

---

## 11. Non-goals and anti-patterns

- No faked media output. Unavailable provider → `deferred`, always.
- No taste score an engine cannot justify. Report what was judged.
- No cloud APIs anywhere in the core or the engines.
- No vector DB. JSON on disk is sufficient for continuity.
- No hardcoded model names outside config.
- No engine-specific vocabulary in the core.
- No `self`-assigned per-run state on shared objects.
- FABRICA does not get pulled in. The `Judge` seam is the only accommodation.
