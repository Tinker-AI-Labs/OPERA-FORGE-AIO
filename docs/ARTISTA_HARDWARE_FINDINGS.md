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

## Standing correction (see also `.notes/posting.txt`)

This machine has exactly one GPU (`lspci`: `03:00.0`, AMD RX 6600). Headless
or separate-TTY launch of ComfyUI does **not** isolate a future GPU hang
from the live desktop session — a `MODE1` device reset hits every client on
that GPU (compute job and compositor alike) regardless of which TTY or
session launched the job that triggered it. Do not treat "run it headless"
as a safety measure on this box; it isn't one.
