"""MUSICA producers.

Two rules shape this file:

* An audio producer stamps the arrangement plan it rendered from into
  ``artifact.content``, and the audio file path into ``artifact.path``. The
  judge reviews the former and measures the latter, and says which is which.
* ``available`` reflects the environment. No soundfont, no fluidsynth, no
  ACE-Step -> the task defers. Silence is never rendered as "output".
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Sequence

from opera.config import RoleConfig
from opera.errors import ProducerError, ProducerUnavailable
from opera.protocols import LLMClient
from opera.schemas import Artifact, Brief

COMPOSER_SYSTEM = (
    "You are a composer. Produce a concrete arrangement plan: key, tempo, time "
    "signature, section structure with bar counts, and instrumentation per "
    "section. Be specific and playable. Output only the plan."
)

ARRANGER_SYSTEM = (
    "You are an arranger. Turn the arrangement plan into a precise, renderable "
    "specification -- exact parts, voicings and dynamics per section. Output only "
    "the specification."
)


class LLMPlanProducer:
    """Writes an arrangement plan. Its product is text, and it says so."""

    def __init__(self, client: LLMClient, role: str, config: RoleConfig,
                 *, system: str = COMPOSER_SYSTEM) -> None:
        if client is None:
            raise ProducerError(f"role {role!r} requires an explicit LLM client")
        self.client = client
        self.name = role
        self.role = role
        self.config = config
        self.kind = config.kind
        self.system = config.system_prompt or system
        self.available = True

    async def produce(self, brief: Brief) -> Artifact:
        parts = []
        if brief.context.strip():
            parts.append(f"Project context:\n{brief.context.strip()}\n")
        parts.append(f"Task: {brief.goal.strip()}")
        if brief.is_revision and brief.prior is not None:
            parts.append(f"\nYour previous plan:\n{brief.prior.content}")
            if brief.issues:
                parts.append("\nAddress every one of these:\n" +
                             "\n".join(f"- {i}" for i in brief.issues))
            parts.append("\nReturn the full revised plan.")
        try:
            text = await self.client.complete(
                system=self.system, prompt="\n".join(parts), model=self.config.model,
                timeout=self.config.timeout_s, no_think=self.config.no_think,
                options={"temperature": self.config.temperature},
            )
        except Exception as exc:
            raise ProducerUnavailable(f"role {self.role!r} could not reach its model: {exc}") from exc
        if not text.strip():
            raise ProducerError(f"role {self.role!r} returned an empty plan")
        return Artifact(kind=self.kind, content=text.strip(), producer=self.name,
                        meta={"model": self.config.model, "role": self.role})


class _SubprocessRenderer:
    """Shared machinery for renderers that shell out to a local binary."""

    def __init__(self, role: str, config: RoleConfig, *, binary: str,
                 output_dir: str | Path, available: bool | None = None) -> None:
        self.name = role
        self.role = role
        self.config = config
        self.kind = config.kind
        self.binary = binary
        self.output_dir = Path(output_dir)
        self.available = self._probe() if available is None else available

    def _probe(self) -> bool:
        return shutil.which(self.binary) is not None

    def _out_path(self, brief: Brief) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{brief.task_id}_a{brief.attempt}_{uuid.uuid4().hex[:6]}.wav"

    async def _run(self, cmd: Sequence[str]) -> subprocess.CompletedProcess:
        return await asyncio.to_thread(
            subprocess.run, list(cmd), capture_output=True, text=True,
            timeout=self.config.timeout_s,
        )

    @staticmethod
    def _plan_text(brief: Brief) -> str:
        """The plan this render came from -- the judge reviews exactly this."""
        if plan := brief.params.get("plan"):
            return str(plan)
        if brief.prior is not None and brief.prior.kind == "text":
            return brief.prior.content
        parts = [brief.goal.strip()]
        if brief.context.strip():
            parts.append(brief.context.strip())
        if brief.issues:
            parts.append("Corrections requested: " + "; ".join(brief.issues))
        return "\n\n".join(parts)


class FluidSynthProducer(_SubprocessRenderer):
    """Renders a MIDI file to WAV with a soundfont.

    Needs both ``fluidsynth`` on PATH and a readable soundfont; missing either
    makes it unavailable rather than silent.
    """

    def __init__(self, role: str, config: RoleConfig, *, soundfont: str | Path,
                 output_dir: str | Path, binary: str = "fluidsynth",
                 sample_rate: int = 44100, available: bool | None = None) -> None:
        self.soundfont = Path(soundfont)
        self.sample_rate = sample_rate
        super().__init__(role, config, binary=binary, output_dir=output_dir,
                         available=available)

    def _probe(self) -> bool:
        return shutil.which(self.binary) is not None and self.soundfont.exists()

    async def produce(self, brief: Brief) -> Artifact:
        midi = brief.params.get("midi_path")
        if not midi or not Path(midi).exists():
            raise ProducerUnavailable(
                f"role {self.role!r} needs a MIDI file to render; none was supplied"
            )
        out = self._out_path(brief)
        proc = await self._run([
            self.binary, "-ni", "-F", str(out), "-r", str(self.sample_rate),
            str(self.soundfont), str(midi),
        ])
        if proc.returncode != 0 or not out.exists():
            raise ProducerError(f"fluidsynth failed: {proc.stderr.strip()[:300]}")
        return Artifact(
            kind=self.kind,
            content=self._plan_text(brief),   # what the judge will review
            path=str(out),
            producer=self.name,
            meta={"renderer": "fluidsynth", "soundfont": str(self.soundfont),
                  "midi_path": str(midi), "sample_rate": self.sample_rate},
        )


class AceStepProducer(_SubprocessRenderer):
    """Renders audio with a local ACE-Step CLI.

    The command is a configurable template rather than a guess at anyone's
    argument order: ``{prompt}``, ``{output}`` and ``{duration}`` are substituted.
    """

    DEFAULT_TEMPLATE = ("{binary}", "--prompt", "{prompt}", "--output", "{output}",
                        "--duration", "{duration}")

    def __init__(self, role: str, config: RoleConfig, *, output_dir: str | Path,
                 binary: str = "ace-step", template: Sequence[str] | None = None,
                 duration_s: float = 30.0, available: bool | None = None) -> None:
        self.template = tuple(template or self.DEFAULT_TEMPLATE)
        self.duration_s = duration_s
        super().__init__(role, config, binary=binary, output_dir=output_dir,
                         available=available)

    async def produce(self, brief: Brief) -> Artifact:
        plan = self._plan_text(brief)
        out = self._out_path(brief)
        duration = float(brief.params.get("duration_s", self.duration_s))
        cmd = [
            part.format(binary=self.binary, prompt=plan, output=str(out),
                        duration=f"{duration:g}")
            for part in self.template
        ]
        proc = await self._run(cmd)
        if proc.returncode != 0 or not out.exists():
            raise ProducerError(f"ace-step failed: {proc.stderr.strip()[:300]}")
        return Artifact(
            kind=self.kind,
            content=plan,
            path=str(out),
            producer=self.name,
            meta={"renderer": "ace-step", "target": {"duration_s": duration},
                  "command": cmd[0]},
        )
