"""Spec 5.2 / 5.3 -- the parsing bug that presents as poor model quality."""

import json

import pytest

from opera.errors import JSONParseError
from opera.llm.parsing import extract_json, extract_json_object, strip_code_fences, strip_think
from opera.llm.stub import think_wrapped


def test_think_wrapped_json_parses():
    """The headline case: braces inside the reasoning trace must not be spanned."""
    payload = {"score": 0.8, "passed": True, "issues": []}
    raw = think_wrapped(payload)
    assert extract_json(raw) == payload


def test_think_block_containing_braces_is_not_spanned():
    raw = '<think>maybe {"score": 0.1} is right</think>\n{"score": 0.9}'
    assert extract_json(raw)["score"] == 0.9


def test_unclosed_think_block_raises_rather_than_guessing():
    with pytest.raises(JSONParseError):
        extract_json('<think>I am still thinking about {"score"')


def test_close_tag_without_open_tag():
    """Some builds emit the trace bare and only tag its end."""
    raw = 'thinking out loud {"nope": 1}</think>{"score": 0.5, "passed": false}'
    assert extract_json(raw)["score"] == 0.5


def test_code_fence_stripped():
    raw = '```json\n{"a": 1}\n```'
    assert strip_code_fences(raw) == '{"a": 1}'
    assert extract_json(raw) == {"a": 1}


def test_fence_inside_think_block():
    raw = '<think>hmm</think>\n```json\n{"tasks": []}\n```'
    assert extract_json(raw) == {"tasks": []}


def test_prose_wrapped_json_falls_back_to_brace_scan():
    raw = 'Sure! Here is the plan:\n{"tasks": [{"goal": "x"}]}\nHope that helps.'
    assert extract_json(raw)["tasks"][0]["goal"] == "x"


def test_brace_scan_respects_string_literals():
    raw = 'preamble {"text": "a } brace in a string", "n": 2} trailing'
    assert extract_json(raw)["n"] == 2


def test_escaped_quote_inside_string():
    payload = {"text": 'she said "hi" }', "n": 3}
    raw = f"noise {json.dumps(payload)} noise"
    assert extract_json(raw) == payload


def test_json_array_is_extracted():
    assert extract_json('<think>x</think>[1, 2, 3]') == [1, 2, 3]


def test_failure_is_structured_not_none():
    """Spec 5.3: a parse failure carries the evidence, it does not return None."""
    with pytest.raises(JSONParseError) as exc:
        extract_json("no structure here at all", stage="planner")
    assert exc.value.raw == "no structure here at all"
    assert exc.value.stage == "planner"


def test_empty_after_stripping_is_an_error():
    with pytest.raises(JSONParseError):
        extract_json("<think>only reasoning, no answer</think>")


def test_extract_json_object_rejects_arrays():
    with pytest.raises(JSONParseError):
        extract_json_object("[1,2]")


def test_strip_think_leaves_clean_text_alone():
    assert strip_think("just an answer") == "just an answer"
