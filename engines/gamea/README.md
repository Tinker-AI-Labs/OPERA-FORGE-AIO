# engines/gamea/ — GAMEA as a real OPERA engine

Wraps the existing bridge/webui (Meshroom + Blender + ComfyUI) behind
the Producer/Judge shape ARTISTA and VIDEA already use, so it drops into
`opera/loop.py`'s produce → judge → revise cycle and can use `best_of_n` /
threshold-gating the same way ARTISTA's retoucher does.

## Files

- `clients.py` — `ComfyUiClient` (submit/poll/download against ComfyUI's
  own HTTP API) and `BridgeClient` (wraps `bridge/server.py`'s Meshroom
  + Blender job endpoints). This is where the ComfyUI → filesystem
  handoff gets wired — `download_output_file()` resolves the `/view`
  URL to a local path, closing the gap the bridge README flagged
  ("Clean Up button will prompt you for the path manually until that's
  wired properly").
- `mesh_qc.py` — objective, trimesh-based mesh inspection: tri count,
  watertightness, UV presence, degenerate-face ratio, bounding box
  extent. No opinion, no model call — same "verifiable output" category
  that excluded FABRICA from the shared aesthetic-judge core.
- `judge.py` — `MeshJudge`, a weighted rubric over those checks. Two
  rules carried over from the 2026-08-18 defect fixes: `passed` is
  computed by the judge from its own score, never trusted from upstream;
  no single check is a hard gate except a genuinely unparseable file or
  a zero-extent mesh (both "doesn't exist," not "is mediocre").
- `producer.py` — `GameaProducer.produce(brief) -> Artifact`. Two paths:
  text prompt → ComfyUI workflow (`hunyuan3d`/`triposr`/`sd-texture`,
  same files the existing `workflows/README.md` already documents) or a
  photo folder → Meshroom. Either path always finishes with the Blender
  cleanup pass before returning.
- `spec.py` — the `EngineSpec` that actually registers GAMEA with
  `opera.registry` (not in the original drop-in; see item 3 below).

## Reconciled against the real opera/ core

This was originally written from documented architecture, not the actual
source (the tarball's own words: "Claude Code should check these against
the real files before wiring it in"). It has since been checked and wired
in for real:

1. **Task/Artifact/Verdict field names.** `producer.py` and `judge.py` now
   import the real `opera.schemas.{Task,Artifact,Verdict}` and
   `opera.protocols.{Producer,Judge}` -- no stand-in dataclasses. Producers
   see a `Brief`, not a `Task` (ARTISTA's `ComfyUIProducer` is the working
   precedent); the prompt comes from `brief.params["prompt"]` or
   `brief.goal`, `images_dir`/`workflow` from `brief.params`, and the
   produced file lives at `artifact.path` (not `file_path`).
2. **Structural test.** `tests/core/test_registry.py`'s
   `test_core_declares_no_engine_vocabulary` now bans `"foundry"` (GAMEA's
   role name) alongside VIDEA/ARTISTA/MUSICA's own role words -- the same
   pattern the other three already use (engine *names* were never banned,
   only role vocabulary).
3. **Router entry.** Not added to `opera/router.py` -- that module is
   deliberately engine-agnostic ("nothing in this module names a role")
   and there is no cross-engine content router anywhere in the real
   source; which engine runs is chosen explicitly (`--engine`/`engine:`
   field), not sniffed from prompt text. `engines/gamea/spec.py` (new,
   not in the original drop-in) supplies its own `ROUTER_KEYWORDS` for its
   `foundry` role and registers with `opera.registry`, exactly like
   ARTISTA/VIDEA/MUSICA's own spec.py files.
4. **Shared ComfyUI client.** Checked: ARTISTA's `ComfyUIProducer` has its
   own submit/poll/download sequence, but it's private and inline, bound
   to ARTISTA's own hardcoded node ids and image-only outputs -- not a
   reusable class. `ComfyUiClient` stays in `clients.py`, rewritten to
   async httpx (matching ARTISTA's approach) instead of the tarball's
   blocking `requests` + `time.sleep`, since `GameaProducer.produce()` is
   itself async.
5. **config.py.** `clients.py` no longer reads env vars. Workflow/output
   directories and the two service URLs come from the `foundry` role's
   `RoleConfig.options` (see `spec.py`) -- the existing generic per-role
   settings dict -- rather than new dedicated fields added to the shared
   `RoleConfig` dataclass every other engine's roles use too.

## What's actually tested here

`tests/engines/test_gamea_judge.py` — 5 tests against synthetic trimesh
primitives (box, oversized icosphere, unreadable file, degenerate
bounding box). No Blender/Meshroom/ComfyUI needed to run these; all 5
pass against the stub. `producer.py`'s ComfyUI/Meshroom/Blender paths
are NOT covered by a test here — they need your actual machine (the
same three-step manual test order the bridge README already lays out:
bridge alone, Blender standalone, Meshroom standalone) before a live
produce→judge run makes sense.

## Not touched

`server.py`, `blender_cleanup.py`, `index.html`, `requirements.txt` —
left exactly as they were. `GameaProducer` calls into `server.py`'s
existing endpoints rather than replacing any of it.
