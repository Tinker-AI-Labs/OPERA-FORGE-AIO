"""Spec 5.1 / 5.4 / 5.5 -- structured output, explicit clients, bounded retries."""

import json

import httpx
import pytest

from opera.config import LLMConfig
from opera.errors import LLMResponseError, LLMTransportError
from opera.llm.ollama import OllamaClient
from opera.llm.stub import ScriptedLLMClient, StubLLMClient


def _client(handler, **cfg):
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    return OllamaClient(LLMConfig(backoff_base_s=0.0, backoff_max_s=0.0, **cfg), client=http)


def _ok(content="hello"):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


async def test_format_json_is_requested_not_hoped_for():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _ok('{"a":1}')

    client = _client(handler)
    await client.complete(prompt="p", format_json=True)
    assert seen["format"] == "json"


async def test_format_json_absent_when_not_requested():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _ok()

    await _client(handler).complete(prompt="p")
    assert "format" not in seen


async def test_no_think_appended_to_system_prompt():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _ok()

    await _client(handler).complete(prompt="p", system="You plan.", no_think=True)
    system = seen["messages"][0]["content"]
    assert system.endswith("/no_think")


async def test_no_think_not_duplicated():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _ok()

    await _client(handler).complete(prompt="p", system="You plan. /no_think", no_think=True)
    assert seen["messages"][0]["content"].count("/no_think") == 1


async def test_images_ride_on_the_user_message():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _ok()

    await _client(handler).complete(prompt="describe", images=["AAAA"])
    assert seen["messages"][-1]["images"] == ["AAAA"]


async def test_think_block_stripped_from_completion():
    client = _client(lambda r: _ok('<think>hmm</think>\nthe answer'))
    assert await client.complete(prompt="p") == "the answer"


async def test_transport_errors_retry_then_raise_transport_error():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("refused", request=request)

    client = _client(handler, max_retries=3)
    with pytest.raises(LLMTransportError) as exc:
        await client.complete(prompt="p")
    assert calls["n"] == 3
    assert exc.value.attempts == 3


async def test_transport_error_recovers_within_budget():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("refused", request=request)
        return _ok("recovered")

    client = _client(handler, max_retries=3)
    assert await client.complete(prompt="p") == "recovered"
    assert calls["n"] == 3


async def test_5xx_is_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="overloaded") if calls["n"] == 1 else _ok("ok")

    assert await _client(handler, max_retries=3).complete(prompt="p") == "ok"
    assert calls["n"] == 2


async def test_4xx_is_not_retried():
    """A missing model does not appear because we asked three times."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404, text="model not found")

    with pytest.raises(LLMResponseError) as exc:
        await _client(handler, max_retries=3).complete(prompt="p")
    assert calls["n"] == 1
    assert exc.value.status == 404


async def test_empty_completion_is_an_error_not_empty_string():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": ""}})

    with pytest.raises(LLMResponseError):
        await _client(handler).complete(prompt="p")


async def test_generate_response_shape_supported():
    def handler(request):
        return httpx.Response(200, json={"response": "from generate"})

    assert await _client(handler).complete(prompt="p") == "from generate"


async def test_available_false_when_host_down():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    assert await _client(handler).available() is False


async def test_stub_routes_match_in_order():
    stub = StubLLMClient([("plan", "PLANNED"), ("judge", "JUDGED")], default="DEFAULT")
    assert await stub.complete(prompt="please plan this") == "PLANNED"
    assert await stub.complete(prompt="please judge this") == "JUDGED"
    assert await stub.complete(prompt="something else") == "DEFAULT"
    assert len(stub.calls) == 3


async def test_scripted_client_raises_when_exhausted():
    """A silent wrap-around would hide an unexpected extra call."""
    scripted = ScriptedLLMClient(["one"])
    assert await scripted.complete(prompt="x") == "one"
    with pytest.raises(AssertionError):
        await scripted.complete(prompt="x")
