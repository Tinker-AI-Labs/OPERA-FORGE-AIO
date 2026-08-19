"""Shared wiring for the CLI and the API.

One place decides how an engine, a client and a store are assembled, so the two
surfaces cannot drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opera import registry
from opera.bible import ProjectStore
from opera.config import OperaConfig, load_config
from opera.errors import ConfigError
from opera.llm.ollama import OllamaClient
from opera.llm.stub import StubLLMClient
from opera.planner import LLMPlanner
from opera.protocols import LLMClient
from opera.registry import EngineSpec
from opera.runner import Runner

import engines  # noqa: F401  -- registers the in-tree engines

STUB_PLAN = json.dumps({"tasks": [{"goal": "Draft the work", "role": None, "kind": None}]})
STUB_VERDICT = json.dumps({"score": 0.9, "passed": True, "issues": []})


def stub_client() -> StubLLMClient:
    """A client that makes a full run complete without any model host.

    Selected only by an explicit ``--stub``; never as a fallback (spec 5.4).
    """
    return StubLLMClient(
        [
            (r"Reply with JSON only, in this exact shape", STUB_PLAN),
            (r"Reply with JSON only", STUB_VERDICT),
        ],
        default="[stub output -- no model was called]",
    )


@dataclass
class AppContext:
    config: OperaConfig
    store: ProjectStore
    client: LLMClient
    engine_name: str
    stub: bool = False

    def spec(self, **kwargs: Any) -> EngineSpec:
        return registry.get(self.engine_name, client=self.client, llm=self.config.llm, **kwargs)

    def runner(self, spec: EngineSpec | None = None) -> Runner:
        spec = spec or self.spec()
        # The stub path goes through the real planner too, so --stub exercises
        # the same code the live path does rather than a shortcut around it.
        planner = LLMPlanner(self.client, spec, model=self.config.llm.planner_model)
        return Runner(spec, self.store, planner=planner, config=self.config)

    async def aclose(self) -> None:
        close = getattr(self.client, "aclose", None)
        if close is not None:
            await close()


def build_context(
    *,
    engine: str,
    stub: bool = False,
    config_path: str | Path | None = None,
    projects_dir: str | Path | None = None,
) -> AppContext:
    config = load_config(config_path)
    if projects_dir:
        config = config.with_overrides(projects_dir=Path(projects_dir).expanduser())
    if engine.lower() not in registry.available():
        raise ConfigError(f"unknown engine {engine!r}; available: {registry.available()}")
    client: LLMClient = stub_client() if stub else OllamaClient(config.llm)
    return AppContext(
        config=config,
        store=ProjectStore(config.projects_dir, config),
        client=client,
        engine_name=engine.lower(),
        stub=stub,
    )
