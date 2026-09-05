"""
GameaProducer -- turns a Brief into a cleaned-up, textured .glb Artifact.

Mirrors ARTISTA's Producer abstraction: produce(brief) -> Artifact, no
opera/ core code needs to know a single thing about Meshroom, Blender,
or Hunyuan3D.

Two entry paths, matching index.html's existing prompt-vs-photos modes:
  - text prompt  -> ComfyUI (Hunyuan3D-2 / TripoSR / SDXL workflow)
  - photo folder -> Meshroom (via the local bridge) for photogrammetry
Either path always finishes with the Blender cleanup pass (normals fix,
voxel remesh, decimate-if-heavy, export .glb) before returning.

RECONCILED (checked against the real source, not assumed):
  - Task/Artifact are opera/schemas.py's real pydantic models, not stand-in
    dataclasses. Producers see a ``Brief`` (opera/protocols.py's Producer
    protocol), not a ``Task`` -- ARTISTA's ComfyUIProducer is the working
    precedent. The text prompt comes from ``brief.params["prompt"]`` when a
    caller supplies one explicitly, else ``brief.goal`` -- same precedence
    ARTISTA's ComfyUIProducer._prompt_from() uses. ``images_dir`` and
    ``workflow`` have no dedicated Brief field either (Brief is engine-
    agnostic, spec 3.1) so they read from ``brief.params`` the same way.
    Artifact's real field is ``path``, not ``file_path``.
  - Workflow JSON loading/prompt-injection duplicates index.html's
    injectPrompt() convention (overwrite any CLIPTextEncode node's `text`
    field) -- there is no shared opera util for this to import instead (see
    clients.py's docstring for why ARTISTA's own version isn't reusable).
  - Workflow directory, output directory, and the two service URLs no
    longer come from env vars -- they resolve from the ``foundry`` role's
    ``RoleConfig.options`` (see engines/gamea/spec.py), which is the real,
    existing per-role settings dict (opera/config.py). New dedicated
    dataclass fields were deliberately *not* added to the shared RoleConfig
    for these -- that type is used by every engine's roles, and workflow
    directories/bridge URLs are GAMEA-only vocabulary (spec 3.3).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Optional

import httpx

from opera.config import RoleConfig
from opera.errors import ProducerError, ProducerUnavailable
from opera.schemas import Artifact, Brief

from .clients import BridgeClient, ComfyUiClient, GameaClientError, GameaJobError

DEFAULT_WORKFLOW = "hunyuan3d"


def _load_workflow(workflows_dir: str, name: str) -> dict:
    path = os.path.join(workflows_dir, f"{name}.json")
    if not os.path.isfile(path):
        raise ProducerError(
            f"workflow file not found: {path} -- export it from ComfyUI's "
            f"'Save (API Format)' menu, per workflows/README.md"
        )
    with open(path) as f:
        return json.load(f)


def _inject_prompt(workflow: dict, prompt: str) -> dict:
    """Same convention as index.html's injectPrompt(): overwrite the
    text input on any node whose class_type contains CLIPTextEncode."""
    for node in workflow.values():
        if isinstance(node, dict) and "CLIPTextEncode" in node.get("class_type", ""):
            node.setdefault("inputs", {})["text"] = prompt
    return workflow


def _find_output_file_ref(history_entry: dict) -> tuple[str, str, str]:
    """Pull (filename, subfolder, type) for the first saved output file
    out of a ComfyUI /history entry's outputs block."""
    outputs = history_entry.get("outputs", {})
    for node_output in outputs.values():
        for key in ("images", "gltfs", "files"):
            items = node_output.get(key)
            if items:
                item = items[0]
                return item["filename"], item.get("subfolder", ""), item.get("type", "output")
    raise ProducerError(f"no output file reference found in history entry: {history_entry}")


class GameaProducer:
    """produce(brief) -> Artifact, satisfying opera.protocols.Producer the
    same way ARTISTA's ComfyUIProducer does."""

    def __init__(
        self,
        role: str,
        config: RoleConfig,
        *,
        comfy_client: Optional[ComfyUiClient] = None,
        bridge_client: Optional[BridgeClient] = None,
        available: Optional[bool] = None,
    ) -> None:
        self.name = role
        self.role = role
        self.config = config
        self.kind = config.kind
        opts = config.options
        self.workflows_dir = opts.get("workflows_dir", "workflows")
        self.output_dir = opts.get("output_dir", "gamea_output")
        self.comfy = comfy_client or ComfyUiClient(
            base_url=opts.get("comfy_url", "http://127.0.0.1:8188")
        )
        self.bridge = bridge_client or BridgeClient(
            base_url=opts.get("bridge_url", "http://127.0.0.1:5050")
        )

        # Availability is a fact about the environment, not an aspiration
        # (ARTISTA's ComfyUIProducer.probe() sets the same precedent). Every
        # task -- prompt or photo path -- ends with the bridge's Blender
        # cleanup step, so the bridge being reachable is the one
        # precondition every task shares; that is what gets probed.
        self.available = self.probe() if available is None else available

    def probe(self) -> bool:
        try:
            with httpx.Client(timeout=3.0) as c:
                resp = c.get(f"{self.bridge.base_url}/status")
                return resp.status_code == 200 and bool(resp.json().get("blender_found"))
        except httpx.HTTPError:
            return False

    async def produce(self, brief: Brief) -> Artifact:
        job_id = uuid.uuid4().hex[:8]

        images_dir = brief.params.get("images_dir")
        prompt = str(brief.params.get("prompt") or brief.goal).strip()
        workflow_name = brief.params.get("workflow", DEFAULT_WORKFLOW)

        if images_dir:
            raw_path = await self._produce_from_photos(images_dir)
            source = "meshroom"
        else:
            raw_path = await self._produce_from_prompt(prompt, workflow_name, job_id)
            source = f"comfyui:{workflow_name}"

        cleaned = await self._cleanup(raw_path)
        output_path = cleaned.get("output_path")
        if not output_path:
            raise ProducerError(f"blender cleanup completed but produced no output_path: {cleaned}")

        return Artifact(
            kind=self.kind,
            path=output_path,
            producer=self.name,
            meta={
                "source": source,
                "job_id": job_id,
                "raw_path": raw_path,
                **({"prompt": prompt} if not images_dir else {"images_dir": images_dir}),
                "cleanup_stats": cleaned.get("stats", {}),
            },
        )

    async def _produce_from_prompt(self, prompt: str, workflow_name: str, job_id: str) -> str:
        workflow = _load_workflow(self.workflows_dir, workflow_name)
        workflow = _inject_prompt(workflow, prompt)

        try:
            prompt_id = await self.comfy.submit_workflow(workflow)
            history_entry = await self.comfy.wait_for_output(prompt_id)
        except GameaJobError as exc:
            raise ProducerError(f"ComfyUI generation failed: {exc}") from exc
        except GameaClientError as exc:
            raise ProducerUnavailable(f"ComfyUI is unreachable: {exc}") from exc
        filename, subfolder, folder_type = _find_output_file_ref(history_entry)

        os.makedirs(self.output_dir, exist_ok=True)
        dest = os.path.join(self.output_dir, f"{job_id}_raw.glb")
        try:
            await self.comfy.download_output_file(filename, subfolder, folder_type, dest)
        except GameaClientError as exc:
            raise ProducerUnavailable(f"ComfyUI is unreachable: {exc}") from exc
        return dest

    async def _produce_from_photos(self, images_dir: str) -> str:
        try:
            job = await self.bridge.run_meshroom(images_dir)
        except GameaJobError as exc:
            raise ProducerError(f"Meshroom photogrammetry failed: {exc}") from exc
        except GameaClientError as exc:
            raise ProducerUnavailable(f"the local bridge is unreachable: {exc}") from exc

        raw_path = job.get("output_path")
        if not raw_path:
            raise ProducerError(f"meshroom job completed but produced no output_path: {job}")
        return raw_path

    async def _cleanup(self, raw_path: str) -> dict:
        try:
            return await self.bridge.run_blender_cleanup(raw_path)
        except GameaJobError as exc:
            raise ProducerError(f"Blender cleanup failed: {exc}") from exc
        except GameaClientError as exc:
            raise ProducerUnavailable(f"the local bridge is unreachable: {exc}") from exc
