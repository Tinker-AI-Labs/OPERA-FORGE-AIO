"""Deterministic in-process LLM clients for tests and ``--stub`` runs.

These are never selected implicitly. The prototype defaulted to a stub when no
client was passed, which turns a wiring mistake into plausible fake output
(spec 5.4) -- here the caller has to ask for it by name.
"""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Any, Callable, Iterable

Handler = Callable[[str, str], str]
Route = tuple[str, str | Handler]


class StubLLMClient:
    """Matches routes against ``system + prompt`` and returns canned output.

    Routes are ordered; the first whose pattern is found (case-insensitive
    substring, or regex when compiled) wins.
    """

    name = "stub"

    def __init__(
        self,
        routes: Iterable[Route] | None = None,
        *,
        default: str | Handler = "stub output",
        record: bool = True,
    ) -> None:
        self.routes: list[Route] = list(routes or [])
        self.default = default
        self.record = record
        self.calls: list[dict[str, Any]] = []

    def route(self, pattern: str, response: str | Handler) -> "StubLLMClient":
        self.routes.append((pattern, response))
        return self

    async def available(self) -> bool:
        return True

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
    ) -> str:
        if self.record:
            self.calls.append(
                {
                    "prompt": prompt,
                    "system": system,
                    "model": model,
                    "format_json": format_json,
                    "images": list(images or []),
                    "no_think": no_think,
                }
            )
        haystack = f"{system}\n{prompt}"
        for pattern, response in self.routes:
            if re.search(pattern, haystack, re.IGNORECASE):
                return response(system, prompt) if callable(response) else response
        return self.default(system, prompt) if callable(self.default) else self.default


class ScriptedLLMClient:
    """Returns a fixed sequence of responses, in order. Raises when exhausted."""

    name = "scripted"

    def __init__(self, responses: Iterable[str], *, cycle: bool = False) -> None:
        self._initial = list(responses)
        self._queue: deque[str] = deque(self._initial)
        self.cycle = cycle
        self.calls: list[dict[str, Any]] = []

    async def available(self) -> bool:
        return True

    async def complete(self, *, prompt: str, system: str = "", **kw: Any) -> str:
        self.calls.append({"prompt": prompt, "system": system, **kw})
        if not self._queue:
            if self.cycle and self._initial:
                self._queue = deque(self._initial)
            else:
                raise AssertionError(
                    f"ScriptedLLMClient exhausted after {len(self.calls)} calls; "
                    "the code under test made more LLM calls than the script covers"
                )
        return self._queue.popleft()


def json_response(payload: Any) -> str:
    return json.dumps(payload)


def think_wrapped(payload: Any, musing: str = "Let me consider {\"a\": 1} carefully.") -> str:
    """A qwen3-shaped response: a reasoning trace containing braces, then JSON."""
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return f"<think>\n{musing}\n</think>\n{body}"
