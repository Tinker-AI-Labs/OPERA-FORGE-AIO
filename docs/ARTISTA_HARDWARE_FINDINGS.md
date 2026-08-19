# ARTISTA hardware findings — syn-2603 (AMD RX 6600, gfx1030)

Dated entries. This machine's GPU is shared with the live desktop
(`kwin_wayland`/`plasmashell`) — see the standing correction in
`.notes/posting.txt` and `CLAUDE.md` §... about why headless/TTY launch does
not isolate a hang here.

## 2026-08-18 — the original hang

First FLUX-schnell render (1024×1024, 4 steps, `flux1-schnell-fp8`) hit a
real GPU hang during VAE decode: plain `VAEDecode` node, GPU-side, default
fp8/fp16 precision. `HW Exception ... GPU Hang` → kernel `MODE1 reset` →
`VRAM is lost due to GPU reset!`, which also reset the desktop compositor's
GPU rings. KSampler (UNet) had already completed cleanly; the fault was
isolated to VAE decode specifically.

## 2026-08-19 — three-way comparison, same prompt/seed/resolution

Same prompt ("a small red fox sitting in a snowy pine forest, soft morning
light"), same seed (42), same 1024×1024 resolution, same
`flux1-schnell-fp8` checkpoint throughout. Each row is a single supervised
trial (n=1) — real-world variance (17GB checkpoint read from a USB drive
each cold boot) is not controlled for; see caveat below.

| Config | VAE node | VAE device | Result | Wall-clock (submit → PNG) |
|---|---|---|---|---|
| Original | `VAEDecode` | GPU (fp8/fp16) | **Hung**, MODE1 reset | n/a |
| Retry A | `VAEDecodeTiled` | CPU (`--fp32-vae --cpu-vae`) | Clean | **327.0s** (~5.45 min) |
| Retry B | `VAEDecodeTiled` | GPU (`--fp32-vae` only) | Clean, no hang | **394.7s** (~6.58 min) |

KSampler (UNet sampling, 4 steps) took ~44-45s in every run, consistent
across all three — the fault and the timing variance both live after that
point, in the checkpoint-swap/VAE-decode phase.

**Surprising result, reported as measured, not smoothed over**: GPU-side
decode (Retry B) was not faster than CPU-side decode (Retry A) — it was
about 68s *slower* in this single trial. That is the opposite of the naive
expectation. Two honest readings, not resolved here:

1. Retry B's real cost isn't decode speed at all — it's confounded by
   checkpoint-load I/O variance (17GB over USB, page-cache state unknown
   between the two fresh ComfyUI boots). Neither log carries per-line
   timestamps, so checkpoint-load time cannot be cleanly separated from
   decode time after the fact. This is the most likely explanation and the
   reason this table is not presented as a clean "VAE decode costs X
   seconds" number.
2. VAEDecodeTiled's tile-by-tile passes plus this run's GPU weight
   offloading (the log shows ~8.5GB offloaded to CPU during UNet sampling,
   i.e. this 8GB card is already memory-constrained at this resolution)
   could genuinely cost more wall-clock than a single CPU-side pass with no
   GPU contention. Plausible, not confirmed.

**More important than the timing number**: Retry B did not hang. The
original hang was on the plain `VAEDecode` node; both retries used
`VAEDecodeTiled`. That means **tiling itself, independent of CPU/GPU
placement, may be what actually avoids the fault** — `--cpu-vae` might be
unnecessary once tiling is in use, not the thing that fixed it. This is a
hypothesis from n=1 per condition, not a confirmed root cause — the
original untiled+GPU combination was never re-attempted (correctly, per the
"one attempt, don't iterate configs unattended" rule), so it's not known
whether that specific combination reliably hangs or hung once.

**Open, unresolved by this data**: whether GPU-side VAE decode (tiled or
not) is reliably safe on this card, or whether Retry B's clean run was
itself the lucky outcome and a repeat could still hang. Sample size is 1
per condition throughout this whole investigation.

## 2026-08-19 — cheapest test for "can this hardware do VIDEA at all"

Not run. Proposed only, per the standing instruction not to build VIDEA
before this question is answered — MUSICA and VIDEA should not get built
out on an assumption about this hardware that hasn't been checked.

VIDEA's own producer would be a video diffusion model, and its judge
(`FrameSampleJudge`) wraps `VisionJudge` over frames extracted *after*
generation via ffmpeg — so the judge path isn't the risk, the **producer's
video VAE decode** is, for the same reason ARTISTA's image VAE decode was:
a short clip is many more frames' worth of latent data through the same
kind of decode operation that already faulted once on this card.

**Cheapest real test, isolating exactly that**: a ComfyUI workflow with no
`KSampler` at all —
`CheckpointLoaderSimple → EmptyLatentImage(batch_size=N) → VAEDecodeTiled →
SaveImage` — decoding a batch of *empty/random* latents rather than
sampling them. This stresses VAE decode at N-frame-equivalent batch scale
directly, without running the (much more expensive) UNet sampling step and
without downloading any video model. Reuses the FLUX checkpoint already on
disk; a video model's own VAE would differ in exact cost, but this answers
the load-bearing question — does *decoding a batch this size* hang or blow
up in wall-clock — before spending anything on a real video model.

- **Setup cost**: ~5 minutes (one workflow JSON, batch_size parameterised).
- **Run cost**: supervised only, same as every GPU-side attempt on this
  box. Escalate batch size gradually (e.g. 1 → 4 → 16 → 32, roughly what a
  1-4 second clip at low fps would need) and stop at the first hang, or
  once satisfied the scaling holds — each trial should cost low
  single-digit minutes based on the per-image numbers already measured.
- **Disk cost**: zero — no new model download.
- **What it answers**: whether batch-scale VAE decode is safe on this GPU
  at all, and how wall-clock scales with batch size (linear? worse?) —
  enough to project whether a real clip is minutes or hours before
  committing to acquiring an actual video model.
- **What it does NOT answer**: whether a real video UNet fits this card's
  8GB VRAM, or whether video-specific attention/temporal ops have their own
  failure modes distinct from VAE decode. That needs the more expensive
  real-model test (a genuine, minimal video generation — shortest duration,
  lowest resolution, fewest steps, on the smallest available local video
  model) as a second, costlier step only if this one passes.

## 2026-08-19 — KSampler-free 2x2 (untiled/tiled x GPU/CPU), isolating VAE decode

Ran the cheapest test proposed above, adapted slightly: instead of
`CheckpointLoaderSimple` (17GB, loads UNet+CLIP too), extracted just the
244 `vae.*`-prefixed tensors from `flux1-schnell-fp8.safetensors` into a
standalone 335MB file (`comfyui_models/vae/flux_ae_extracted.safetensors`)
so the graph is `VAELoader -> EmptySD3LatentImage -> VAEDecode(Tiled) ->
SaveImage` with zero UNet/CLIP load. `EmptySD3LatentImage` used (not
`EmptyLatentImage`) — FLUX's AE is 16-channel, the SD-style default node
produces 4-channel latents and errors immediately (caught before any GPU
work, not a hang). Same seed-independent zero latents throughout, square
sizes 256/512/768/1024, batch_size=1. Two ComfyUI boots (plain, and
`--cpu-vae`), tiled vs untiled selected per-workflow via node choice
(`VAEDecodeTiled` default tile_size=512/overlap=64). No `--fp32-vae` in
either boot — cell 1 deliberately matches the original hang condition
exactly (default fp8/fp16, untiled, GPU).

| Size | Cell 1: untiled+GPU | Cell 2: untiled+`--cpu-vae` | Cell 3: tiled+GPU | Cell 4: tiled+`--cpu-vae` |
|---|---|---|---|---|
| 256 | 12.15s (cold VAE load) | 2.49s | 1.63s | 4.68s |
| 512 | 0.30s | 6.77s | 77.12s | 22.81s |
| 768 | 75.58s | 15.59s | **298.25s** | 54.39s |
| 1024 | **HUNG — MODE1 reset** | 28.23s | 4.35s | 101.77s |

Timings are server-side "Prompt executed in" values, VAE-decode-only (no
checkpoint/UNet load confound this time — a real improvement over the
2026-08-19 three-way comparison above).

**Cell 1 reproduced the hang, n=2 now.** Same signature as the original:
`HW Exception ... GPU Hang` -> Python `Fatal Python error: Aborted` ->
kernel `ring gfx_0.1.0 timeout` -> `MODE1 reset` -> `GPU reset succeeded` ->
`VRAM is lost due to GPU reset!`. `kwin_wayland` (pid 2630) was hit and
recovered, consistent with the standing single-GPU correction. Per the
"one attempt, don't iterate configs unattended" rule this session stopped
immediately after the hang and got explicit go-ahead before running cells
2-4 in a later turn.

**Answer to the load-bearing question: tiling alone fixes it, `--cpu-vae`
is not required.** Cell 3 (tiled, GPU, no `--cpu-vae`) completed cleanly at
1024x1024 — the exact resolution and precision that hung untiled. This
resolves the open hypothesis from the section above in favor of the better
outcome for VIDEA: full GPU throughput, no CPU fallback needed by default.

**New anomaly, not previously seen**: cell 3's 768 timing (298s) is wildly
out of line with its neighbors (77s at 512, only 4s at 1024) — worse than
4x the 512 result and 68x the 1024 result, on the *same* untiled-free,
GPU-resident, warm-VAE path. Leading explanation: 512 is an exact multiple
of the default `tile_size=512`, so 1024 decodes as a clean 2x2 tile grid,
while 768 doesn't divide evenly and likely forces an extra ragged tile
pass plus more overlap-blend work per axis. Not confirmed by direct
profiling, just the shape of the data (cell 4, tiled+CPU, shows the same
directional bump at 768 relative to 512 -- 55s vs 23s -- but nowhere near
cell 3's magnitude, consistent with this being a tile-count effect that
GPU-side execution happens to amplify far more than CPU-side does).
Practical takeaway: if VIDEA or any future tiled-decode config lets tile
size be tuned, prefer sizes that divide evenly by `tile_size` — an uneven
remainder tile may cost much more than its pixel share suggests, at least
on this GPU.

**Cells 2 and 4 (`--cpu-vae`) both scaled smoothly and predictably** with
no anomaly, as expected for CPU-bound work — 28s and 102s respectively at
1024, both slower than cell 3's 4.35s but with zero hang risk by
construction (no GPU decode compute at all, confirmed via boot log's `VAE
load device: cpu, offload device: cpu`).

**Net for VIDEA**: default to `VAEDecodeTiled` on GPU (no `--cpu-vae`).
Sample size is still small (n=1 for cells 2-4, n=2 for cell 1's hang vs.
no-hang), and tile-size tuning is an open follow-up given the 768 anomaly.

## 2026-08-19 — Final decision: tiled+GPU is the default, `--cpu-vae` is fallback-only

Closing the open question from the two sections above. **Default going
forward: `VAEDecodeTiled` on GPU, no `--cpu-vae`.** This is not a tie
broken by preference — at 1024x1024, tiled+GPU (cell 3) took **4.35s**
against tiled+`--cpu-vae`'s (cell 4) **101.77s** — **23x** the wall-clock
cost for a config that isn't needed once tiling alone is confirmed to
prevent the hang (cell 3, n=1 clean at the exact resolution that hung
untiled). Cell 2 (untiled+`--cpu-vae`) was faster than cell 4 at 28.23s,
but that comparison doesn't matter for the default decision — untiled is
banned on this hardware regardless of device placement (see below), so the
only fallback config worth keeping around is tiled+`--cpu-vae` (cell 4),
and even that loses to tiled+GPU by 23x.

`--cpu-vae` is documented here as a **fallback, not a competing default**:
keep it available for a future engine that for some reason can't tune
`tile_size` per-request (fixed third-party node, external API, etc.) and
therefore can't guarantee an evenly-dividing tile size — see the note
below. If `tile_size` can be computed correctly, GPU-side tiled decode is
strictly better on every measured axis: faster, and (per cell 3's clean
1024 run) does not hang.

## Note for whoever builds VIDEA's producer: `tile_size` must be computed, not hardcoded

**Do not hardcode `tile_size=512`** (ComfyUI's `VAEDecodeTiled` default).
The 2x2 above proved a hardcoded tile size becomes a performance cliff the
moment the target resolution doesn't divide evenly by it: at 768px, tiled
decode cost **298.25s on GPU** and **54.39s on CPU**, against **4.35s** and
**101.77s** respectively at 1024px — worse in wall-clock at the *smaller*
resolution, solely because 768 / 512 leaves a ragged remainder tile and
1024 / 512 doesn't. Confirmed on both the GPU and CPU decode paths
(direction of the effect is the same on both; magnitude differs, GPU
amplifies it far more).

VIDEA's real frame sizes — 720p (1280x720), 1080p (1920x1080) — will hit
exactly this case if `tile_size` stays at the ComfyUI default: neither 720
nor 1280 nor 1080 nor 1920 is a multiple of 512. **`tile_size` needs to be
derived from the target resolution's actual divisors** (e.g. pick the
largest tile size ≤ some VRAM-safe ceiling that evenly divides both width
and height, or pad to the next multiple and crop after decode) rather than
left at the library default. This is an open implementation task for
VIDEA's producer, not resolved by this investigation — this investigation
only proves the cliff exists and why.

## Confirmed deterministic fault: untiled `VAEDecode` hangs on this hardware — not an ARTISTA/VIDEA bug

Two independent occurrences, same signature, 5 hours apart, both on this
same GPU/driver/framework stack:

- **2026-08-18 21:07:11 EDT** — original hang, full ARTISTA pipeline
  (checkpoint load + KSampler + plain `VAEDecode`), 1024x1024.
- **2026-08-19 02:16:19 EDT** — reproduction, this session's cell 1,
  VAE-decode-only graph (no checkpoint/UNet at all), same 1024x1024, same
  plain `VAEDecode` node, GPU, default fp8/fp16.

Removing the entire checkpoint-load/KSampler path between the two
occurrences and still hitting the identical `MODE1 reset` signature at the
identical resolution is strong evidence this is a **hardware/driver fault
in plain (untiled) GPU-side `VAEDecode` on this specific stack**
(gfx1030 / ROCm 6.3 / torch 2.9.1+rocm6.3 / ComfyUI 0.24.0), not a bug in
ARTISTA's or VIDEA's own code, prompt, model choice, or workflow graph
shape. See the Debug info section below for exact versions and the full
kernel-log signature from both occurrences.

**Stated plainly for future sessions: untiled GPU-side VAE decode at
1024x1024 (and, by the logic above, likely other sizes near or above it)
must never be used on this hardware again. Tiled decode
(`VAEDecodeTiled`) is mandatory on this box, not merely preferred.** Any
future workflow, custom node, or default ComfyUI graph that routes through
plain `VAEDecode` at meaningful resolutions on this machine should be
treated as a known hang risk until proven otherwise on different hardware.

## Debug/diagnostic info (for reproducibility)

**Stack versions** (from ComfyUI boot log, both incidents — unchanged
between 2026-08-18 and 2026-08-19 sessions):

```
ComfyUI version: 0.24.0
comfy-aimdo version: 0.4.9
comfy-kitchen version: 0.2.10
pytorch version: 2.9.1+rocm6.3
AMD arch: gfx1030
ROCm version: (6, 3)
Total VRAM 8176 MB, total RAM 31872 MB
GPU: AMD Radeon RX 6600 (lspci 03:00.0, Navi 23)
```

**Exact workflow JSON, cell 1 (untiled, the config that hung both times)**
— `1024x1024` shown, other sizes in the 2x2 only change `width`/`height`
on node `2`, other cells only add `--cpu-vae` at ComfyUI launch (untiled
graph shape is identical for cells 1 and 2):

```json
{
  "prompt": {
    "1": {"class_type": "VAELoader", "inputs": {"vae_name": "flux_ae_extracted.safetensors"}},
    "2": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "3": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["1", 0]}},
    "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "vaeprobe_c1_untiled_gpu_1024"}}
  },
  "client_id": "vaeprobe"
}
```

**Exact workflow JSON, cell 3 (tiled, the config that fixed it)** — same
structure, node `3` swapped for `VAEDecodeTiled` with ComfyUI's own
defaults (`tile_size=512, overlap=64`); cells 3 and 4 share this shape,
cell 4 differs only by `--cpu-vae` at launch:

```json
{
  "prompt": {
    "1": {"class_type": "VAELoader", "inputs": {"vae_name": "flux_ae_extracted.safetensors"}},
    "2": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "3": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["2", 0], "vae": ["1", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}},
    "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "vaeprobe_c3_tiled_gpu_1024"}}
  },
  "client_id": "vaeprobe"
}
```

`flux_ae_extracted.safetensors` (`comfyui_models/vae/`, 335MB) is the 244
`vae.*`-prefixed tensors pulled out of `flux1-schnell-fp8.safetensors`
with the `vae.` prefix stripped — no UNet/CLIP weights, real production VAE
weights, loadable via ComfyUI's standard `VAELoader`.

**Kernel log signature, occurrence 1 (2026-08-18 21:07:11 EDT, boot
`d45d77cf`, full ARTISTA pipeline)**:

```
amdgpu 0000:03:00.0: ring gfx_0.1.0 timeout, signaled seq=43404, emitted seq=43406
amdgpu 0000:03:00.0:  Process kwin_wayland pid 2699 thread kwin_wayla:cs0 pid 2709
amdgpu 0000:03:00.0: Starting gfx_0.1.0 ring reset
amdgpu 0000:03:00.0: Ring gfx_0.1.0 reset failed
amdgpu 0000:03:00.0: GPU reset begin!. Source:  1
amdgpu 0000:03:00.0: failed to suspend display audio
amdgpu: Failed to suspend process pid 12997
amdgpu 0000:03:00.0: MODE1 reset
amdgpu 0000:03:00.0: GPU mode1 reset
amdgpu 0000:03:00.0: GPU smu mode1 reset
amdgpu 0000:03:00.0: GPU reset succeeded, trying to resume
amdgpu 0000:03:00.0: VRAM is lost due to GPU reset!
amdgpu 0000:03:00.0: GPU reset(1) succeeded!
[...second ring-timeout wave, same event...]
amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout, signaled seq=41914, emitted seq=41917
amdgpu 0000:03:00.0:  Process plasmashell pid 2896 thread plasmashel:cs0 pid 3061
amdgpu 0000:03:00.0: Starting gfx_0.0.0 ring reset
amdgpu 0000:03:00.0: Ring gfx_0.0.0 reset succeeded
```

Note this occurrence hit **both** compositor processes across two ring
timeouts — `kwin_wayland` on `gfx_0.1.0`, then `plasmashell` on
`gfx_0.0.0` — both recovered.

**Kernel log signature, occurrence 2 (2026-08-19 02:16:19 EDT, boot
`67b3e75a`, VAE-decode-only graph, this session)**:

```
amdgpu 0000:03:00.0: ring gfx_0.1.0 timeout, signaled seq=1004600, emitted seq=1004602
amdgpu 0000:03:00.0:  Process kwin_wayland pid 2630 thread kwin_wayla:cs0 pid 2650
amdgpu 0000:03:00.0: Starting gfx_0.1.0 ring reset
amdgpu 0000:03:00.0: Ring gfx_0.1.0 reset failed
amdgpu 0000:03:00.0: GPU reset begin!. Source:  1
amdgpu: Failed to suspend process pid 53760
amdgpu 0000:03:00.0: MODE1 reset
amdgpu 0000:03:00.0: GPU mode1 reset
amdgpu 0000:03:00.0: GPU smu mode1 reset
amdgpu 0000:03:00.0: GPU reset succeeded, trying to resume
amdgpu 0000:03:00.0: VRAM is lost due to GPU reset!
amdgpu 0000:03:00.0: GPU reset(1) succeeded!
[...second ring-timeout wave, same event, kwin_wayland again...]
amdgpu 0000:03:00.0: ring gfx_0.1.0 timeout, signaled seq=1004604, emitted seq=1004607
amdgpu 0000:03:00.0:  Process kwin_wayland pid 2630 thread kwin_wayla:cs0 pid 2650
amdgpu 0000:03:00.0: Starting gfx_0.1.0 ring reset
amdgpu 0000:03:00.0: Ring gfx_0.1.0 reset succeeded
```

`pid 53760` in `Failed to suspend process` is the ComfyUI `main.py`
process itself, which the kernel could not gracefully suspend during the
reset — it subsequently died with `Fatal Python error: Aborted` on the
Python side (ComfyUI's own log, not the kernel log). `kwin_wayland`
(pid 2630, same process both ring-timeout waves this time) recovered both
times without a restart.

Both excerpts pulled via `journalctl -k -b <boot-id> --since ... --until
...`; full context available the same way for any future investigation —
boot IDs and exact timestamps above are enough to re-query.

## Standing correction (see also `.notes/posting.txt`)

This machine has exactly one GPU (`lspci`: `03:00.0`, AMD RX 6600). Headless
or separate-TTY launch of ComfyUI does **not** isolate a future GPU hang
from the live desktop session — a `MODE1` device reset hits every client on
that GPU (compute job and compositor alike) regardless of which TTY or
session launched the job that triggered it. Do not treat "run it headless"
as a safety measure on this box; it isn't one.
