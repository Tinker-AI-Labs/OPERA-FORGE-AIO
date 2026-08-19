"""Async Ollama client.

Requests structured output rather than hoping for it, strips reasoning traces,
and separates transport retries from review-driven revisions (spec 5).
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from ..config import LLMConfig
from ..errors import LLMResponseError, LLMTransportError
from .parsing import strip_think

_NO_THINK = "/no_think"


class OllamaClient:
    """Talks to a local Ollama host. No cloud, no fallback to a stub."""

    name = "ollama"

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or LLMConfig()
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.config.host)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def available(self) -> bool:
        try:
            resp = await self._http().get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def models(self) -> list[str]:
        resp = await self._http().get("/api/tags", timeout=10.0)
        resp.raise_for_status()
        return [m.get("name", "") for m in resp.json().get("models", [])]

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
        """One completion. Retries transport failures only."""
        model = model or self.config.default_model
        timeout = timeout or self.config.default_timeout_s

        if no_think and system and _NO_THINK not in system:
            system = f"{system.rstrip()}\n{_NO_THINK}"
        elif no_think and not system:
            system = _NO_THINK

        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            # Ollama takes base64 payloads (no data: prefix) on the message.
            user_message["images"] = images

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(user_message)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": dict(options or {}),
        }
        if format_json:
            # Spec 5.1: ask for JSON, do not scrape braces out of prose.
            payload["format"] = "json"

        text = await self._post_with_retries("/api/chat", payload, timeout)
        return strip_think(text) if self.config.strip_think else text

    async def _post_with_retries(self, path: str, payload: dict[str, Any], timeout: float) -> str:
        attempts = self.config.max_retries
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = await self._http().post(path, json=payload, timeout=timeout)
            except httpx.HTTPError as exc:
                last = exc
                if attempt == attempts:
                    break
                await asyncio.sleep(self._backoff(attempt))
                continue

            if resp.status_code >= 500:
                last = LLMResponseError(
                    f"ollama returned {resp.status_code}", status=resp.status_code, body=resp.text[:500]
                )
                if attempt == attempts:
                    break
                await asyncio.sleep(self._backoff(attempt))
                continue
            if resp.status_code >= 400:
                # Client errors are not retryable -- a missing model will not
                # appear because we asked three times.
                raise LLMResponseError(
                    f"ollama returned {resp.status_code}", status=resp.status_code, body=resp.text[:500]
                )

            data = resp.json()
            content = (data.get("message") or {}).get("content", "")
            if not content and "response" in data:  # /api/generate shape
                content = data["response"]
            if not content:
                raise LLMResponseError("ollama returned an empty completion", status=resp.status_code)
            return content

        raise LLMTransportError(
            f"ollama unreachable after {attempts} attempts: {last}",
            attempts=attempts,
            cause=last if isinstance(last, Exception) else None,
        )

    def _backoff(self, attempt: int) -> float:
        delay = min(self.config.backoff_base_s * (2 ** (attempt - 1)), self.config.backoff_max_s)
        return delay * (0.5 + random.random() / 2)  # jitter
