"""Goal -> (role, kind) routing. Deterministic by default (spec 7).

Two corrections over the prototype are structural here, not incidental:

1. **Text roles are scored before media roles.** The prototype checked media
   keywords first, so *"write a script for a video clip"* routed wholly to media
   and produced no writing.
2. **Bare media words do not classify.** ``art``, ``score`` and ``music`` match
   "state of the art" and "a musical score plays in the background". A media
   role's single-word keyword only counts when an explicit generation verb is
   present; multi-word phrases stand on their own.

The vocabulary is engine-supplied. Nothing in this module names a role.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import RoleConfig
from .llm.parsing import extract_json_object
from .protocols import LLMClient

DEFAULT_MEDIA_KINDS = frozenset({"image", "video", "audio", "model", "render"})

# An explicit request to *make* something, as opposed to a passing mention.
DEFAULT_GENERATION_VERBS = (
    "generate", "render", "draw", "illustrate", "paint", "compose", "animate",
    "synthesize", "synthesise", "produce", "create", "make", "design", "shoot",
    "storyboard", "record",
)

_ROUTER_SYSTEM = (
    "You route a single creative goal to exactly one role. "
    "Reply with JSON only: {\"role\": \"<role>\", \"reason\": \"<short reason>\"}. "
    "Choose only from the roles listed by the user."
)


@dataclass(frozen=True)
class RouteResult:
    role: str
    kind: str
    reason: str
    score: float = 0.0
    fallback: bool = False


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    parts = [re.escape(p) for p in phrase.strip().split()]
    return re.compile(r"\b" + r"\W+".join(parts) + r"\w*\b", re.IGNORECASE)


@dataclass
class KeywordRouter:
    """Rule-based router over an engine's keyword table."""

    keywords: dict[str, list[str]]
    roles: dict[str, RoleConfig]
    default_role: str
    default_kind: str = "text"
    media_kinds: frozenset[str] = DEFAULT_MEDIA_KINDS
    generation_verbs: tuple[str, ...] = DEFAULT_GENERATION_VERBS
    _compiled: dict[str, list[tuple[re.Pattern[str], int]]] = field(init=False, default_factory=dict)
    _verb_re: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        for role, words in self.keywords.items():
            self._compiled[role] = [
                (_phrase_pattern(w), len(w.strip().split())) for w in words if w.strip()
            ]
        self._verb_re = re.compile(
            r"\b(" + "|".join(re.escape(v) for v in self.generation_verbs) + r")\w*\b",
            re.IGNORECASE,
        )

    def is_media_role(self, role: str) -> bool:
        cfg = self.roles.get(role)
        return bool(cfg and cfg.kind in self.media_kinds)

    def kind_for(self, role: str) -> str:
        cfg = self.roles.get(role)
        return cfg.kind if cfg else self.default_kind

    def _score(self, goal: str, role: str, *, has_verb: bool) -> float:
        total = 0.0
        media = self.is_media_role(role)
        for pattern, words in self._compiled.get(role, []):
            if not pattern.search(goal):
                continue
            if media and words == 1 and not has_verb:
                # Fix 2: a bare media noun is not a request to generate media.
                continue
            total += float(words)  # multi-word phrases are stronger evidence
        return total

    def route(self, goal: str) -> RouteResult:
        goal = goal or ""
        has_verb = bool(self._verb_re.search(goal))

        text_roles = [r for r in self._compiled if not self.is_media_role(r)]
        media_roles = [r for r in self._compiled if self.is_media_role(r)]

        # Fix 1: text roles are considered first and win outright when they match.
        for group, label in ((text_roles, "text"), (media_roles, "media")):
            scored = [(self._score(goal, r, has_verb=has_verb), r) for r in group]
            scored = [(s, r) for s, r in scored if s > 0]
            if scored:
                scored.sort(key=lambda pair: (-pair[0], pair[1]))
                score, role = scored[0]
                return RouteResult(
                    role=role,
                    kind=self.kind_for(role),
                    reason=f"{label} keyword match (score {score:g})",
                    score=score,
                )

        return RouteResult(
            role=self.default_role,
            kind=self.kind_for(self.default_role),
            reason="no keyword matched; engine default",
            fallback=True,
        )


class LLMRouter:
    """Opt-in fallback, consulted only when the keyword pass is ambiguous."""

    def __init__(self, keyword_router: KeywordRouter, client: LLMClient, *, model: str,
                 timeout_s: float = 60.0, no_think: bool = True) -> None:
        self.keyword_router = keyword_router
        self.client = client
        self.model = model
        self.timeout_s = timeout_s
        self.no_think = no_think

    def route(self, goal: str) -> RouteResult:
        """Synchronous surface, matching KeywordRouter. Falls back to keywords."""
        return self.keyword_router.route(goal)

    async def aroute(self, goal: str) -> RouteResult:
        rule = self.keyword_router.route(goal)
        if not rule.fallback:
            return rule
        roles = ", ".join(sorted(self.keyword_router.roles))
        try:
            raw = await self.client.complete(
                system=_ROUTER_SYSTEM,
                prompt=f"Roles: {roles}\n\nGoal: {goal}",
                model=self.model,
                format_json=True,
                timeout=self.timeout_s,
                no_think=self.no_think,
            )
            data = extract_json_object(raw, stage="router")
        except Exception:
            # A router that cannot be reached must not take the run down;
            # the deterministic result already in hand is the safe answer.
            return rule
        role = str(data.get("role", "")).strip()
        if role not in self.keyword_router.roles:
            return rule
        return RouteResult(
            role=role,
            kind=self.keyword_router.kind_for(role),
            reason=f"llm router: {str(data.get('reason', ''))[:120]}",
        )
