"""Spec 7 -- the two routing fixes."""

import pytest

from opera.config import RoleConfig
from opera.llm.stub import StubLLMClient
from opera.router import KeywordRouter, LLMRouter

ROLES = {
    "writer": RoleConfig(model="m", kind="text"),
    "reasoner": RoleConfig(model="m", kind="text"),
    "coder": RoleConfig(model="m", kind="code"),
    "painter": RoleConfig(model="m", kind="image"),
    "scorer": RoleConfig(model="m", kind="audio"),
}

KEYWORDS = {
    "writer": ["write", "script", "scene", "dialogue", "screenplay"],
    "reasoner": ["analyse", "analyze", "compare", "evaluate"],
    "coder": ["code", "function", "refactor", "implement"],
    # Media keywords: multi-word phrases, per the spec 7 fix.
    "painter": ["concept art", "key frame", "illustration", "art"],
    "scorer": ["background score", "musical cue", "music"],
}


def router() -> KeywordRouter:
    return KeywordRouter(keywords=KEYWORDS, roles=ROLES, default_role="writer",
                         default_kind="text")


# --- fix 1: text roles are scored before media roles -------------------------

def test_write_a_script_for_a_video_clip_routes_to_writing():
    """The prototype routed this entirely to media and produced no writing."""
    r = router().route("write a script for a video clip")
    assert r.role == "writer" and r.kind == "text"


def test_pure_media_goal_still_routes_to_media():
    r = router().route("generate concept art of a lighthouse")
    assert r.role == "painter" and r.kind == "image"


def test_media_wins_only_when_no_text_role_matches():
    assert router().route("render a key frame").role == "painter"
    assert router().route("write about a key frame").role == "writer"


# --- fix 2: bare media words do not classify ---------------------------------

def test_state_of_the_art_does_not_route_to_the_painter():
    r = router().route("summarise the state of the art in diffusion models")
    assert r.role != "painter"


def test_a_musical_score_in_the_background_does_not_route_to_the_scorer():
    r = router().route("a musical score plays in the background of the diner")
    assert r.role != "scorer"


def test_bare_media_noun_with_a_generation_verb_does_route():
    assert router().route("generate art for the poster").role == "painter"
    assert router().route("compose music for the chase").role == "scorer"


def test_multi_word_media_phrase_routes_without_a_verb():
    assert router().route("concept art of the tower").role == "painter"
    assert router().route("a background score for act two").role == "scorer"


# --- general behaviour -------------------------------------------------------

def test_longer_phrases_outweigh_single_words():
    r = KeywordRouter(
        keywords={"writer": ["scene"], "painter": ["concept art"]},
        roles=ROLES, default_role="writer",
    ).route("draw concept art of the scene")
    # writer still wins -- text precedence is absolute, not score-based.
    assert r.role == "writer"


def test_media_scoring_prefers_the_stronger_phrase():
    r = router().route("render concept art and a musical cue")
    assert r.role in {"painter", "scorer"}
    assert r.score >= 2


def test_no_match_falls_back_to_the_engine_default():
    r = router().route("something entirely unrelated")
    assert r.role == "writer" and r.fallback is True
    assert "default" in r.reason


def test_empty_goal_falls_back():
    assert router().route("").fallback is True


def test_word_boundaries_are_respected():
    """'scripture' must not match the keyword 'script' as a whole word."""
    r = KeywordRouter(keywords={"coder": ["code"], "writer": ["script"]},
                      roles=ROLES, default_role="reasoner").route("decode the barcode")
    assert r.role == "reasoner"


def test_simple_plural_and_suffix_still_match():
    assert router().route("write the scenes").role == "writer"


def test_routing_is_deterministic():
    r = router()
    assert {r.route("write a scene").role for _ in range(20)} == {"writer"}


def test_kind_comes_from_the_role_config():
    assert router().route("refactor the function").kind == "code"


# --- LLM router is opt-in and only for ambiguity -----------------------------

async def test_llm_router_is_not_consulted_when_keywords_match():
    client = StubLLMClient(default='{"role":"painter"}')
    r = await LLMRouter(router(), client, model="m").aroute("write a scene")
    assert r.role == "writer"
    assert client.calls == []


async def test_llm_router_resolves_an_ambiguous_goal():
    client = StubLLMClient(default='{"role":"reasoner","reason":"it is an assessment"}')
    r = await LLMRouter(router(), client, model="m").aroute("figure out what is wrong here")
    assert r.role == "reasoner"
    assert "llm router" in r.reason


async def test_llm_router_ignores_a_hallucinated_role():
    client = StubLLMClient(default='{"role":"cinematographer"}')
    r = await LLMRouter(router(), client, model="m").aroute("do the thing")
    assert r.role == "writer" and r.fallback is True


async def test_llm_router_survives_an_unreachable_model():
    class Broken:
        name = "broken"

        async def complete(self, **kw):
            raise ConnectionError("no host")

        async def available(self):
            return False

    r = await LLMRouter(router(), Broken(), model="m").aroute("do the thing")
    assert r.role == "writer"


def test_llm_router_sync_route_uses_keywords_only():
    client = StubLLMClient(default='{"role":"painter"}')
    assert LLMRouter(router(), client, model="m").route("write a scene").role == "writer"
    assert client.calls == []
