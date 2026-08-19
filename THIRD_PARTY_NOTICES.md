# Third-Party Notices

OPERA is proprietary to TINKER-VERSE LLC (see LICENSE). This file records the
licensing posture of everything OPERA depends on or talks to, so the question
"can this ship commercially?" has a written answer rather than a guess.

Verify each entry against the installed version before relying on it. Update
this file whenever a dependency or backend is added.

---

## Python dependencies (linked / imported)

These are imported directly into the OPERA process, so their terms apply to
distribution of OPERA itself. All are permissive and impose no copyleft
obligation.

| Package        | License      | Notes                                    |
|----------------|--------------|------------------------------------------|
| FastAPI        | MIT          | `service` extra. Attribution only         |
| Starlette      | BSD-3-Clause | Transitive dependency of FastAPI. Attribution only |
| uvicorn        | BSD-3-Clause | `service` extra, imported directly in `service/cli.py`'s `serve` command. Attribution only |
| Pydantic       | MIT          | Attribution only                          |
| httpx          | BSD-3-Clause | Attribution only                          |
| pytest         | MIT          | `dev` extra. Test-time only, not distributed |
| pytest-asyncio | Apache-2.0   | `dev` extra. Test-time only, not distributed |

No GPL, LGPL, AGPL, or source-available dependency is linked into OPERA.
**This is a deliberate constraint. Adding one would change what the Company
can do with the Software — treat it as a licensing decision, not a
dependency decision.**

---

## Backends invoked over the network (not linked)

OPERA communicates with these over HTTP. They run as separate processes and
are not linked, imported, bundled, or redistributed. Their licenses govern
those programs, not OPERA.

| Backend  | License      | Interface                     |
|----------|--------------|-------------------------------|
| Ollama   | MIT          | HTTP API                      |
| ComfyUI  | GPL-3.0      | HTTP API (`/prompt`, `/history`, `/view`) |

**On ComfyUI specifically:** GPL-3.0 is copyleft, but arms-length process
separation over a network API is not linking, and OPERA neither bundles nor
distributes ComfyUI. OPERA's proprietary status is not affected. This
separation must be preserved — importing ComfyUI code into OPERA, vendoring
it, or shipping it inside an OPERA distribution would raise a genuine
copyleft question that does not exist today.

If ComfyUI ever needs to be shipped alongside OPERA, get legal advice first.

---

## Models and weights

Model licenses are separate from software licenses and frequently more
restrictive. They govern both use and, in some cases, the commercial use of
generated output.

| Model              | License              | Commercial use |
|--------------------|----------------------|----------------|
| FLUX.1 [schnell]   | Apache-2.0           | Permitted      |
| FLUX.1 [dev]       | FLUX.1 Non-Commercial| **Not permitted** |
| llava              | Apache-2.0 (check weights) | Verify per build |
| qwen2.5-vl         | Qwen license         | Verify tier    |
| ACE-Step           | Verify before use    | Unverified     |

**FLUX.1 [dev] is a live trap.** It is non-commercial. ARTISTA currently
targets FLUX.1 [schnell], which is Apache-2.0 and commercially usable.
Swapping to [dev] for quality would make commercial output infringing.
Any change to the default checkpoint is a licensing decision.

Verify every model's terms before it becomes a default in any engine.

---

## Evaluated and rejected

| Project | License | Outcome |
|---------|---------|---------|
| Maestro (Blizaine) | WanGP Non-Commercial Evaluation License 1.1 | Rejected 2026-08-18 as an OPERA frontend. Commercial use of the software, including hosted services and APIs, requires a separate license from the WanGP licensor. |
| Donjon  | Non-commercial | Previously rejected for the dungeon generator on the same grounds. |

---

*This file is a working record, not legal advice.*
