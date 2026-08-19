"""The three seams OPERA is built on.

The loop talks to Producer and Judge only. It never imports a concrete
reviewer class -- that is the single accommodation left for FABRICA, so a
deterministic verifier could be substituted later without touching loop.py
(spec 1, 11).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .schemas import Artifact, Brief, Task, Verdict


@runtime_checkable
class Producer(Protocol):
    """Anything that makes an artifact.

    An LLM agent and a ComfyUI image generator are both Producers. There is no
    second code path for media.
    """

    name: str
    kind: str
    available: bool

    async def produce(self, brief: Brief) -> Artifact: ...


@runtime_checkable
class Judge(Protocol):
    name: str

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict: ...


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface over a local model host.

    ``format_json`` maps to Ollama's ``"format": "json"`` -- structured output is
    requested, not scraped out of prose (spec 5.1).
    """

    name: str

    async def complete(
        self,
        *,
        prompt: str,
        system: str = "",
        model: str | None = None,
        format_json: bool = False,
        images: list[str] | None = None,
        timeout: float | None = None,
        options: dict[str, Any] | None = None,
        no_think: bool = False,
    ) -> str: ...

    async def available(self) -> bool: ...


@runtime_checkable
class HumanGate(Protocol):
    """Optional per-engine approval step (MUSICA's human gate, spec 2).

    Returning False parks the task at ``awaiting_review`` rather than claiming
    it is done.
    """

    name: str

    async def approve(self, task: Task, artifact: Artifact) -> bool: ...
