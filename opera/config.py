"""Configuration. No model name is hardcoded anywhere else (spec 11).

Values resolve in this order: explicit argument > JSON config file > environment
> the neutral defaults here. The defaults name models only because *something*
has to be the fallback; every engine overrides them from its own config.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import ConfigError

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class RoleConfig:
    """Per-role model settings. Engine-supplied vocabulary (spec 3.3)."""

    model: str
    system_prompt: str = ""
    timeout_s: float = 180.0
    max_attempts: int = 2
    temperature: float = 0.7
    no_think: bool = False
    pass_threshold: float = 0.7
    kind: str = "text"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMConfig:
    host: str = DEFAULT_OLLAMA_HOST
    default_model: str = "qwen3:8b"
    vision_model: str = "qwen2.5vl:7b"
    planner_model: str = "qwen3:8b"
    judge_model: str = "qwen3:8b"
    router_model: str = "qwen3:8b"
    default_timeout_s: float = 180.0
    max_retries: int = 3
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    # qwen3 and friends emit <think> blocks by default; strip and suppress them
    # for structured roles (spec 5.2).
    strip_think: bool = True
    no_think_roles: tuple[str, ...] = ("planner", "judge", "router")


@dataclass(frozen=True)
class LoopConfig:
    max_attempts: int = 2
    pass_threshold: float = 0.7
    # A revision pass is only worth spending when the judge actually said what
    # was wrong.
    revise_only_with_issues: bool = False


@dataclass(frozen=True)
class OperaConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    projects_dir: Path = field(default_factory=lambda: _default_projects_dir())
    context_token_budget: int = 1200
    context_per_category: int = 6
    ledger_compact_threshold: int = 500

    def with_overrides(self, **kw: Any) -> "OperaConfig":
        return replace(self, **kw)


def _default_projects_dir() -> Path:
    """Resolve the project store root.

    Honours OPERA_HOME, then the float's own path module if it is importable,
    then the user's home. No literal float path appears here (float contract 3).
    """
    env = os.environ.get("OPERA_HOME")
    if env:
        return Path(env).expanduser()
    try:  # pragma: no cover - depends on host layout
        from t1nk3r_paths import P  # type: ignore

        return Path(P.BRAIN).parent / "opera" / "projects"
    except Exception:
        return Path.home() / ".opera" / "projects"


def load_config(path: str | Path | None = None) -> OperaConfig:
    """Load config from JSON, falling back to environment and defaults."""
    cfg = OperaConfig()
    data: dict[str, Any] = {}
    path = path or os.environ.get("OPERA_CONFIG")
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config file is not valid JSON: {p}: {exc}") from exc

    llm_data = dict(data.get("llm", {}))
    if host := os.environ.get("OPERA_OLLAMA_HOST"):
        llm_data.setdefault("host", host)
    if llm_data:
        if "no_think_roles" in llm_data:
            llm_data["no_think_roles"] = tuple(llm_data["no_think_roles"])
        cfg = cfg.with_overrides(llm=replace(cfg.llm, **llm_data))
    if loop_data := data.get("loop"):
        cfg = cfg.with_overrides(loop=replace(cfg.loop, **loop_data))
    if projects_dir := data.get("projects_dir"):
        cfg = cfg.with_overrides(projects_dir=Path(projects_dir).expanduser())
    for key in ("context_token_budget", "context_per_category", "ledger_compact_threshold"):
        if key in data:
            cfg = cfg.with_overrides(**{key: data[key]})
    return cfg


def roles_from_dict(raw: dict[str, dict[str, Any]]) -> dict[str, RoleConfig]:
    """Build a role vocabulary from plain config data."""
    out: dict[str, RoleConfig] = {}
    for name, spec in raw.items():
        if "model" not in spec:
            raise ConfigError(f"role {name!r} has no model configured")
        out[name] = RoleConfig(**spec)
    return out
