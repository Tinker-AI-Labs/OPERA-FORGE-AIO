"""ARTISTA producers.

The point of this file is what is *not* in it: no branch in ``opera.loop``, no
second review path, no changes to the core. A ComfyUI image generator satisfies
the same ``Producer`` protocol an LLM agent does (spec 3.1, 10.7).
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from pathlib import Path
from typing import Any

import httpx

from opera.config import RoleConfig
from opera.errors import ProducerError, ProducerUnavailable
from opera.protocols import LLMClient
from opera.schemas import Artifact, Brief

PROMPT_SYSTEM = (
    "You turn a creative brief into a single image-generation prompt. "
    "Output only the prompt text -- no preamble, no options, no explanation. "
    "Carry through any established style, palette or character details from the "
    "project context."
)


class PromptSmith:
    """A text Producer whose product is the prompt another producer will use.

    Reports ``kind="text"`` honestly: it wrote words, not a picture.
    """

    def __init__(self, client: LLMClient, role: str, config: RoleConfig) -> None:
        if client is None:
            raise ProducerError(f"PromptSmith {role!r} requires an explicit LLM client")
        self.client = client
        self.name = role
        self.role = role
        self.config = config
        self.kind = config.kind
        self.available = True

    async def produce(self, brief: Brief) -> Artifact:
        parts = []
        if brief.context.strip():
            parts.append(f"Project context:\n{brief.context.strip()}\n")
        parts.append(f"Brief: {brief.goal.strip()}")
        if brief.is_revision and brief.prior is not None:
            parts.append(f"\nPrevious prompt:\n{brief.prior.content}")
            if brief.issues:
                parts.append("\nFix these:\n" + "\n".join(f"- {i}" for i in brief.issues))
        try:
            text = await self.client.complete(
                system=self.config.system_prompt or PROMPT_SYSTEM,
                prompt="\n".join(parts),
                model=self.config.model,
                timeout=self.config.timeout_s,
                no_think=self.config.no_think,
                options={"temperature": self.config.temperature},
            )
        except Exception as exc:
            raise ProducerUnavailable(f"role {self.role!r} could not reach its model: {exc}") from exc
        if not text.strip():
            raise ProducerError(f"role {self.role!r} returned an empty prompt")
        return Artifact(kind=self.kind, content=text.strip(), producer=self.name,
                        meta={"model": self.config.model, "role": self.role})


class ComfyUIProducer:
    """Generates an image through a local ComfyUI instance.

    ``available`` is resolved once at construction by ``probe()``. If ComfyUI is
    not running, the task is deferred with a reason -- this producer has no code
    path that returns a placeholder image (spec 11).
    """

    def __init__(
        self,
        role: str,
        config: RoleConfig,
        *,
        host: str = "http://127.0.0.1:8188",
        workflow: dict[str, Any] | None = None,
        workflow_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        prompt_node: str = "6",
        seed_node: str = "3",
        available: bool | None = None,
        client: httpx.AsyncClient | None = None,
        poll_interval_s: float = 1.0,
        timeout_s: float | None = None,
        use_context: bool = True,
        max_prompt_chars: int = 1500,
    ) -> None:
        self.name = role
        self.role = role
        self.config = config
        self.kind = config.kind
        self.host = host.rstrip("/")
        self.prompt_node = prompt_node
        self.seed_node = seed_node
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s or config.timeout_s
        self.use_context = use_context
        self.max_prompt_chars = max_prompt_chars
        self._client = client
        self.output_dir = Path(output_dir) if output_dir else None

        if workflow is None and workflow_path is not None:
            path = Path(workflow_path)
            if not path.exists():
                raise ProducerError(f"comfyui workflow not found: {path}")
            workflow = json.loads(path.read_text())
        self.workflow = workflow

        # Availability is a fact about the environment, not an aspiration.
        self.available = self.probe() if available is None else available

    def probe(self) -> bool:
        if self.workflow is None:
            return False
        try:
            with httpx.Client(timeout=3.0) as c:
                return c.get(f"{self.host}/system_stats").status_code == 200
        except httpx.HTTPError:
            return False

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.host)
        return self._client

    def _build_workflow(self, prompt: str, seed: int) -> dict[str, Any]:
        wf = json.loads(json.dumps(self.workflow))  # deep copy
        if self.prompt_node in wf:
            wf[self.prompt_node].setdefault("inputs", {})["text"] = prompt
        if self.seed_node in wf:
            wf[self.seed_node].setdefault("inputs", {})["seed"] = seed
        return wf

    def _prompt_from(self, brief: Brief) -> str:
        """Build the generation prompt.

        Precedence: an explicitly supplied prompt, then a prior text artifact on
        a revision, then the goal composed with project context. The context
        matters -- in a two-stage run the prompt_smith's output reaches this
        producer through the bible, and dropping it would silently discard the
        earlier task's entire contribution.
        """
        if upstream := brief.params.get("prompt"):
            return str(upstream)
        if brief.is_revision and brief.prior is not None and brief.prior.kind == "text":
            return brief.prior.content

        parts: list[str] = []
        if self.use_context and brief.context.strip():
            parts.append(self._style_from_context(brief.context))
        parts.append(brief.goal.strip())
        if brief.issues:
            parts.append("Corrections: " + "; ".join(brief.issues))
        prompt = ". ".join(p for p in parts if p).strip()
        return prompt[: self.max_prompt_chars]

    @staticmethod
    def _style_from_context(context: str) -> str:
        """Flatten the bible context into prompt-shaped text.

        Section headers and list bullets are structure for a chat model, not
        words a text encoder should see.
        """
        lines = []
        for line in context.splitlines():
            line = line.strip()
            if not line or line.startswith("##"):
                continue
            lines.append(line.lstrip("- ").strip())
        return ". ".join(lines)

    async def produce(self, brief: Brief) -> Artifact:
        if self.workflow is None:
            raise ProducerUnavailable(f"role {self.role!r} has no ComfyUI workflow configured")

        prompt = self._prompt_from(brief)
        seed = uuid.uuid4().int % (2**31)
        http = self._http()

        try:
            resp = await http.post("/prompt",
                                   json={"prompt": self._build_workflow(prompt, seed)},
                                   timeout=30.0)
            resp.raise_for_status()
            prompt_id = resp.json()["prompt_id"]
        except (httpx.HTTPError, KeyError) as exc:
            raise ProducerUnavailable(f"comfyui rejected the job: {exc}") from exc

        image_bytes, filename = await self._await_image(http, prompt_id)

        path = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"{brief.task_id}_{brief.attempt}_{filename}"
            path.write_bytes(image_bytes)

        meta: dict[str, Any] = {
            "prompt": prompt,
            "seed": seed,
            "comfyui_prompt_id": prompt_id,
            "filename": filename,
        }
        if path is None:
            # Only inline the bytes when there is no file to point the judge at.
            # The project JSON is rewritten after every task (spec 4.6), so a
            # base64 image in `meta` would be re-serialised on each save.
            meta["image_b64"] = base64.b64encode(image_bytes).decode("ascii")

        return Artifact(
            kind=self.kind,
            content="",
            path=str(path) if path else None,
            producer=self.name,
            meta=meta,
        )

    async def _await_image(self, http: httpx.AsyncClient, prompt_id: str) -> tuple[bytes, str]:
        deadline = asyncio.get_running_loop().time() + self.timeout_s
        while True:
            if asyncio.get_running_loop().time() > deadline:
                raise ProducerError(f"comfyui job {prompt_id} did not finish in {self.timeout_s}s")
            try:
                hist = await http.get(f"/history/{prompt_id}", timeout=15.0)
                hist.raise_for_status()
                data = hist.json()
            except httpx.HTTPError as exc:
                raise ProducerUnavailable(f"comfyui went away while polling: {exc}") from exc

            entry = data.get(prompt_id)
            if entry:
                for node_output in (entry.get("outputs") or {}).values():
                    for image in node_output.get("images", []):
                        params = {"filename": image["filename"],
                                  "subfolder": image.get("subfolder", ""),
                                  "type": image.get("type", "output")}
                        view = await http.get("/view", params=params, timeout=60.0)
                        view.raise_for_status()
                        return view.content, image["filename"]
                if entry.get("status", {}).get("status_str") == "error":
                    raise ProducerError(f"comfyui job {prompt_id} failed: {entry.get('status')}")
            await asyncio.sleep(self.poll_interval_s)
