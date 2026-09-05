# OPERA

[![tests](https://github.com/Tinker-AI-Labs/OPERA-FORGE-AIO/actions/workflows/tests.yml/badge.svg)](https://github.com/Tinker-AI-Labs/OPERA-FORGE-AIO/actions/workflows/tests.yml)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

A local, engine-agnostic **produce → judge → revise → persist** loop with project
memory. Four FORGE engines are built on top of it.

| Engine | Produces | Judge | `judged` reported |
|---|---|---|---|
| VIDEA | text, code, video | LLM review; frame sampling for video | `artifact` / `frames` |
| ARTISTA | images (ComfyUI) | local vision model | `artifact` (images), `plan` (prompts) |
| MUSICA | audio (ACE-Step / FluidSynth) | deterministic checks + plan review + human gate | `artifact+plan` |
| GAMEA | 3D assets (.glb via Meshroom or ComfyUI, always finished with a Blender cleanup pass) | deterministic mesh QC (trimesh: tri budget, watertightness, UVs, degenerate faces) | `artifact` |

Everything runs locally: Ollama for LLM roles, local generators for media. No
cloud APIs, no vector DB, no web UI.

**FABRICA is out of scope.** The only accommodation is that the loop calls the
`Judge` protocol and never a concrete reviewer class, so a deterministic
verifier could be dropped in later without touching `loop.py`.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[service,dev]'
```

## Use

```bash
opera --stub run "write the opening scene"          # no model host needed
opera health                                         # what is actually reachable
opera --engine musica health                         # per-engine producer status
opera run "a lighthouse film" --name Lighthouse
opera show <run-id> --project <project-id>
opera bible --project <project-id>
opera serve --port 8000
```

`--stub` is an explicit choice. There is no silent fallback to a stub client:
a missing client raises rather than producing plausible fake output.

## Configuration

No model name is hardcoded outside config. Point `--config` at a JSON file, or
set `OPERA_CONFIG`:

```json
{
  "llm": {
    "host": "http://127.0.0.1:11434",
    "default_model": "qwen3:8b",
    "vision_model": "qwen2.5vl:7b",
    "judge_model": "qwen3:8b",
    "max_retries": 3
  },
  "loop": { "max_attempts": 2, "pass_threshold": 0.7 },
  "projects_dir": "~/.opera/projects"
}
```

The project store root resolves from `OPERA_HOME`, then `t1nk3r_paths.P` if
importable, then `~/.opera/projects`. No literal float path appears in any
source file.

## Layout

```
opera/          the shared core -- knows nothing about any engine
  schemas.py    Task, Run, Artifact, Verdict, Project, Bible, Ledger
  protocols.py  Producer, Judge, LLMClient, HumanGate
  llm/          async Ollama client, stub client, defensive JSON parsing
  config.py     RoleConfig / LLMConfig / LoopConfig
  bible.py      project memory (Bible) + telemetry (Ledger) + atomic store
  judges.py     LLMJudge, VisionJudge, FrameSampleJudge, DeterministicJudge,
                CompositeJudge
  loop.py       produce -> judge -> revise, per task
  planner.py    prompt -> task list, plus the validation of it
  router.py     goal -> (role, kind), engine-supplied vocabulary
  runner.py     orchestrates a run over a task list
  registry.py   EngineSpec registration
engines/        videa/ artista/ musica/ gamea/ -- each: spec.py producer(s).py judge.py
service/        api.py (FastAPI) cli.py context.py
tests/          core/ engines/ + acceptance checklist
```

## The rules the loop actually enforces

Each is a test, not a comment.

1. **Every stored artifact carries its own verdict.** The loop is
   produce → judge → (revise → judge)\*. There is no exit path on which a
   revision was produced but not judged.
2. **`max_attempts` counts total productions.** `max_attempts=2` means at most
   two artifacts, one initial plus one revision.
3. **Context is recomputed per task, not per run.** Task 2 sees task 1's output.
4. **A failing task fails that task, not the run.** A producer crash on task 5
   does not discard tasks 1–4.
5. **Planner output is validated, never trusted.** An unknown role falls back to
   the router, an unknown kind to the engine default, and the correction is
   recorded on the task.
6. **Persistence is incremental.** The project is saved after every task.
7. **No shared mutable run state.** Everything a run touches is a local inside
   `run()`. Two concurrent projects cannot cross-write each other's files.

## What OPERA refuses to do

- **Fake media.** An unavailable producer defers the task with a reason and
  produces nothing. There is no placeholder path.
- **Overclaim coverage.** Every `Verdict` states what was actually assessed.
  MUSICA measures the file and reads the plan, so it reports `artifact+plan` —
  it never implies it listened to the music. VIDEA's video judge reports
  `frames` and names what it did not assess.
- **Collapse a mixed run to `done`.** One done task and one deferred task is a
  `partial` run, and the counts say so.
- **Fall back to a stub.** A missing LLM client is a wiring bug and raises.

## Testing

```bash
.venv/bin/python -m pytest tests -q
```

The suite runs with no Ollama, no ComfyUI and no ffmpeg. `tests/test_acceptance.py`
is the spec's own required-case checklist, one test per case, in order.

## Adding an engine

An engine is data. Build an `EngineSpec` and register it:

```python
from opera import registry
from opera.registry import EngineSpec

def build(client, *, llm=None, **kw) -> EngineSpec:
    return EngineSpec(
        name="myengine",
        roles={...},                 # your vocabulary, from config
        producers={...},             # keyed by role name
        judge=...,                   # any Judge
        kinds=frozenset({...}),
        planner_system_prompt="...",
        router_keywords={...},       # media entries multi-word or verb-led
        default_role="...",
    )

registry.register("myengine", build)
```

Nothing in `opera/` needs to change. A test enforces that: the core is scanned
for engine role names and fails if any appear as code.
