"""Defensive extraction of structured data from model output (spec 5.3).

Order matters and is the whole point. A reasoning model emits::

    <think>I should return {"score": 1} maybe...</think>
    {"score": 0.8, "passed": true}

A naive first-brace/last-brace scan spans from the brace *inside* the think
block to the final brace, and fails to parse on every single planner and judge
call. That presents as "the model is bad" and is actually this bug (spec 5.2).
So: think blocks first, then fences, then a whole-string parse, and only then a
depth-aware brace scan.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..errors import JSONParseError

_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think\s*>", re.IGNORECASE)
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
_OPENERS = {"{": "}", "[": "]"}


def strip_think(text: str) -> str:
    """Remove reasoning traces.

    Handles the closing tag even when the opening tag is absent -- several
    models emit the trace bare and only tag its end.
    """
    if not text:
        return ""
    match = None
    for match in _THINK_CLOSE.finditer(text):
        pass
    if match is not None:
        return text[match.end():].strip()
    if _THINK_OPEN.search(text):
        # Unclosed trace: the model never got to an answer. Anything before the
        # tag is preamble, anything after is unterminated reasoning.
        return _THINK_OPEN.split(text, maxsplit=1)[0].strip()
    return text.strip()


def strip_code_fences(text: str) -> str:
    """Unwrap a single fenced block; leave prose containing fences alone."""
    if not text:
        return ""
    m = _FENCE.match(text.strip())
    if m:
        return m.group(1).strip()
    return text.strip()


def _scan_span(text: str) -> str | None:
    """Find the first balanced {...} or [...] span, respecting string literals."""
    start = -1
    opener = ""
    for i, ch in enumerate(text):
        if ch in _OPENERS:
            start, opener = i, ch
            break
    if start < 0:
        return None
    closer = _OPENERS[opener]
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def extract_json(text: str, *, stage: str = "") -> Any:
    """Parse structured output, or raise ``JSONParseError`` carrying the raw text.

    Never returns ``None`` to mean failure -- ``null`` is a legitimate value and
    a silent ``None`` is how a wiring bug becomes a mystery (spec 5.3).
    """
    raw = text or ""
    cleaned = strip_code_fences(strip_think(raw))
    if not cleaned.strip():
        raise JSONParseError("model returned no content after stripping reasoning traces",
                             raw=raw, stage=stage)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    span = _scan_span(cleaned)
    if span is not None:
        try:
            return json.loads(span)
        except json.JSONDecodeError as exc:
            raise JSONParseError(f"balanced JSON span did not parse: {exc}",
                                 raw=raw, stage=stage) from exc
    raise JSONParseError("no JSON object or array found in model output",
                         raw=raw, stage=stage)


def extract_json_object(text: str, *, stage: str = "") -> dict[str, Any]:
    value = extract_json(text, stage=stage)
    if not isinstance(value, dict):
        raise JSONParseError(f"expected a JSON object, got {type(value).__name__}",
                             raw=text or "", stage=stage)
    return value
