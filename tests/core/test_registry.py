"""Spec 3.3 / 11 -- engines are data, and the core stays engine-agnostic."""

import pytest

from opera import registry
from opera.config import RoleConfig
from opera.errors import ConfigError, RegistryError
from opera.registry import EngineSpec
from tests.doubles import ScriptedJudge, ScriptedProducer


def base(**over):
    kwargs = dict(
        name="test",
        roles={"writer": RoleConfig(model="m", kind="text")},
        producers={"writer": ScriptedProducer(name="writer")},
        judge=ScriptedJudge([True]),
        kinds=frozenset({"text"}),
        default_role="writer",
        default_kind="text",
    )
    kwargs.update(over)
    return kwargs


def test_valid_spec_builds():
    s = EngineSpec(**base())
    assert s.fallback_role == "writer" and s.kind_for_role("writer") == "text"


def test_producer_without_a_role_config_is_rejected():
    with pytest.raises(ConfigError, match="no role config"):
        EngineSpec(**base(producers={"writer": ScriptedProducer(), "ghost": ScriptedProducer()}))


def test_role_with_an_undeclared_kind_is_rejected():
    with pytest.raises(ConfigError, match="kind is not declared"):
        EngineSpec(**base(roles={"writer": RoleConfig(model="m", kind="hologram")}))


def test_default_role_without_a_producer_is_rejected():
    with pytest.raises(ConfigError, match="default_role"):
        EngineSpec(**base(default_role="nobody"))


def test_default_kind_must_be_declared():
    with pytest.raises(ConfigError, match="default_kind"):
        EngineSpec(**base(default_kind="video"))


def test_engine_with_no_producers_is_rejected():
    with pytest.raises(ConfigError, match="no producers"):
        EngineSpec(**base(producers={}))


def test_engine_with_no_kinds_is_rejected():
    with pytest.raises(ConfigError, match="no kinds"):
        EngineSpec(**base(kinds=frozenset()))


def test_unknown_role_lookup_is_a_registry_error():
    with pytest.raises(RegistryError):
        EngineSpec(**base()).producer_for("painter")


def test_fallback_role_when_no_default_is_set():
    s = EngineSpec(**base(default_role="",
                          roles={"a": RoleConfig(model="m", kind="text"),
                                 "b": RoleConfig(model="m", kind="text")},
                          producers={"a": ScriptedProducer(name="a"),
                                     "b": ScriptedProducer(name="b")}))
    assert s.fallback_role == "a"


def test_router_is_built_from_the_engine_vocabulary():
    s = EngineSpec(**base(router_keywords={"writer": ["write", "scene"]}))
    assert s.router().route("write a scene").role == "writer"


def test_two_engines_may_use_different_role_vocabularies():
    """`writer/reasoner/coder` is VIDEA's vocabulary, not a global one."""
    videa = EngineSpec(**base(name="videa",
                              roles={"writer": RoleConfig(model="m", kind="text")},
                              producers={"writer": ScriptedProducer(name="writer")}))
    musica = EngineSpec(**base(name="musica",
                               roles={"composer": RoleConfig(model="m", kind="text")},
                               producers={"composer": ScriptedProducer(name="composer")},
                               default_role="composer"))
    assert set(videa.roles) == {"writer"}
    assert set(musica.roles) == {"composer"}


def test_register_and_get(monkeypatch):
    monkeypatch.setattr(registry, "_ENGINES", {})
    registry.register("demo", lambda **kw: EngineSpec(**base(name="demo")))
    assert registry.available() == ["demo"]
    assert registry.get("DEMO").name == "demo"


def test_duplicate_registration_is_rejected(monkeypatch):
    monkeypatch.setattr(registry, "_ENGINES", {})
    registry.register("demo", lambda **kw: EngineSpec(**base()))
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("demo", lambda **kw: EngineSpec(**base()))
    registry.register("demo", lambda **kw: EngineSpec(**base()), replace=True)


def test_unknown_engine_lists_what_is_registered(monkeypatch):
    monkeypatch.setattr(registry, "_ENGINES", {})
    with pytest.raises(RegistryError, match="unknown engine"):
        registry.get("nope")


def test_core_declares_no_engine_vocabulary():
    """Structural guard: no engine's role names may appear as code in the core.

    Scans NAME and STRING tokens but skips comments and docstrings, so the
    prose that *explains* these rules is allowed to name them.
    """
    import io
    import pathlib
    import tokenize

    banned = {"writer", "reasoner", "coder", "composer", "arranger", "mixer",
              "prompt_smith", "retoucher"}
    core = pathlib.Path(__file__).resolve().parents[2] / "opera"
    offenders = []
    for py in sorted(core.rglob("*.py")):
        source = py.read_text()
        docstrings = set()
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        prev_meaningful = tokenize.INDENT
        for tok in tokens:
            if tok.type == tokenize.STRING and prev_meaningful in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT
            ):
                docstrings.add((tok.start, tok.end))
            if tok.type not in (tokenize.COMMENT,):
                prev_meaningful = tok.type
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and (tok.start, tok.end) in docstrings:
                continue
            if tok.type == tokenize.NAME and tok.string in banned:
                offenders.append(f"{py.name}:{tok.start[0]}: {tok.line.strip()}")
            elif tok.type == tokenize.STRING:
                for word in banned:
                    if word in tok.string:
                        offenders.append(f"{py.name}:{tok.start[0]}: {tok.line.strip()}")
                        break
    assert offenders == [], "engine vocabulary leaked into the core:\n" + "\n".join(offenders)


def test_core_has_no_module_level_agent_registry():
    import opera.loop
    import opera.runner

    for module in (opera.loop, opera.runner):
        assert not hasattr(module, "AGENT_REGISTRY")
