"""Engines are data, not forks of the core (spec 3.3).

An ``EngineSpec`` carries an engine's own vocabulary. The core validates against
it at runtime; there is no module-level agent registry and no hardcoded kind
Literal anywhere in ``opera`` (spec 11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .config import RoleConfig
from .errors import ConfigError, RegistryError
from .protocols import HumanGate, Judge, Producer
from .router import DEFAULT_MEDIA_KINDS, KeywordRouter


@dataclass(frozen=True)
class EngineSpec:
    name: str
    roles: dict[str, RoleConfig]
    producers: dict[str, Producer]
    judge: Judge
    kinds: frozenset[str]
    planner_system_prompt: str = ""
    router_keywords: dict[str, list[str]] = field(default_factory=dict)
    default_role: str = ""
    default_kind: str = "text"
    media_kinds: frozenset[str] = DEFAULT_MEDIA_KINDS
    gate: HumanGate | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("EngineSpec needs a name")
        if not self.producers:
            raise ConfigError(f"engine {self.name!r} has no producers")
        if not self.kinds:
            raise ConfigError(f"engine {self.name!r} declares no kinds")
        missing_roles = set(self.producers) - set(self.roles)
        if missing_roles:
            raise ConfigError(
                f"engine {self.name!r} has producers with no role config: {sorted(missing_roles)}"
            )
        bad_kinds = {
            role: cfg.kind for role, cfg in self.roles.items() if cfg.kind not in self.kinds
        }
        if bad_kinds:
            raise ConfigError(
                f"engine {self.name!r} has roles whose kind is not declared: {bad_kinds}"
            )
        if self.default_role and self.default_role not in self.producers:
            raise ConfigError(
                f"engine {self.name!r} default_role {self.default_role!r} has no producer"
            )
        if self.default_kind not in self.kinds:
            raise ConfigError(
                f"engine {self.name!r} default_kind {self.default_kind!r} is not declared"
            )

    @property
    def fallback_role(self) -> str:
        return self.default_role or sorted(self.producers)[0]

    def kind_for_role(self, role: str) -> str:
        cfg = self.roles.get(role)
        if cfg and cfg.kind in self.kinds:
            return cfg.kind
        return self.default_kind

    def producer_for(self, role: str) -> Producer:
        try:
            return self.producers[role]
        except KeyError as exc:
            raise RegistryError(f"engine {self.name!r} has no producer for role {role!r}") from exc

    def role_config(self, role: str) -> RoleConfig | None:
        return self.roles.get(role)

    def router(self) -> KeywordRouter:
        return KeywordRouter(
            keywords=self.router_keywords or {r: [] for r in self.producers},
            roles=self.roles,
            default_role=self.fallback_role,
            default_kind=self.default_kind,
            media_kinds=self.media_kinds,
        )


EngineFactory = Callable[..., EngineSpec]

_ENGINES: dict[str, EngineFactory] = {}


def register(name: str, factory: EngineFactory, *, replace: bool = False) -> None:
    key = name.lower()
    if key in _ENGINES and not replace:
        raise RegistryError(f"engine {name!r} is already registered")
    _ENGINES[key] = factory


def get(name: str, **kwargs) -> EngineSpec:
    key = (name or "").lower()
    if key not in _ENGINES:
        raise RegistryError(f"unknown engine {name!r}; registered: {sorted(_ENGINES) or 'none'}")
    return _ENGINES[key](**kwargs)


def available() -> list[str]:
    return sorted(_ENGINES)


def clear() -> None:  # pragma: no cover - test helper
    _ENGINES.clear()
