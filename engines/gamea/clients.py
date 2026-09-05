"""
Async HTTP clients GameaProducer uses to talk to the two local services.

Neither of these does any generation itself -- they just wrap the two
existing pieces that already work standalone (per bridge/README.md's
test order): ComfyUI's own HTTP API, and the Flask bridge in
bridge/server.py that wraps Meshroom + Blender (neither of which speaks
HTTP natively).

RECONCILED (checked against the real source, not assumed): ARTISTA's
ComfyUIProducer (engines/artista/producers.py) does have its own submit ->
poll /history -> resolve /view sequence, but it is private, inline, and
tailored to ARTISTA's own conventions -- hardcoded prompt/seed node ids
("6"/"3") and only ever looks for an "images" output key. GAMEA needs a
different prompt-injection convention (any node whose class_type contains
CLIPTextEncode, matching index.html's injectPrompt(), since a hunyuan3d/
triposr graph doesn't have a fixed node id to target) and a wider set of
output keys ("images"/"gltfs"/"files", since a 3D save node isn't an image
node). There is no separate, importable client class in ARTISTA to reuse --
only inline methods bound to one producer instance -- so this stays its own
class rather than forking ARTISTA's producer to extract one, which would be
a change to a file this task leaves untouched. It was, however, rewritten
from the tarball's synchronous `requests` + `time.sleep` polling to async
httpx + asyncio.sleep, since GameaProducer.produce() is itself async (the
Producer protocol requires it) and a blocking sleep in an async method would
stall the whole event loop for the duration of a generation job -- the one
respect in which it now matches ARTISTA's approach rather than duplicating
the tarball's.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import httpx


class GameaClientError(RuntimeError):
    """The service could not be reached at all -- a transport failure."""


class GameaJobError(GameaClientError):
    """The service answered, but reported that the job itself failed."""


@dataclass
class ComfyUiClient:
    base_url: str = "http://127.0.0.1:8188"
    poll_interval_s: float = 1.0
    timeout_s: float = 900.0

    async def submit_workflow(self, workflow: dict) -> str:
        """POST a workflow graph (API-format JSON, prompt text already
        injected -- same injectPrompt() convention as index.html) and
        return the prompt_id."""
        try:
            async with httpx.AsyncClient(base_url=self.base_url) as http:
                resp = await http.post("/prompt", json={"prompt": workflow}, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise GameaClientError(f"ComfyUI /prompt unreachable: {exc}") from exc
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise GameaJobError(f"ComfyUI /prompt returned no prompt_id: {data}")
        return prompt_id

    async def wait_for_output(self, prompt_id: str) -> dict:
        """Poll /history/<id> until ComfyUI reports the job done, then
        return the history entry containing output node references."""
        deadline = time.monotonic() + self.timeout_s
        async with httpx.AsyncClient(base_url=self.base_url) as http:
            while time.monotonic() < deadline:
                try:
                    resp = await http.get(f"/history/{prompt_id}", timeout=30.0)
                    resp.raise_for_status()
                    history = resp.json()
                except httpx.HTTPError as exc:
                    raise GameaClientError(f"ComfyUI /history unreachable: {exc}") from exc
                entry = history.get(prompt_id)
                if entry:
                    if entry.get("outputs"):
                        return entry
                    if entry.get("status", {}).get("status_str") == "error":
                        raise GameaJobError(f"ComfyUI job {prompt_id} failed: {entry.get('status')}")
                await asyncio.sleep(self.poll_interval_s)
        raise GameaJobError(f"ComfyUI job {prompt_id} did not finish within {self.timeout_s}s")

    async def download_output_file(
        self, filename: str, subfolder: str, folder_type: str, dest_path: str
    ) -> str:
        """Resolve ComfyUI's /view URL for a produced file and save it
        locally -- the ComfyUI -> filesystem handoff index.html's README
        flags as not wired yet. This is that wiring."""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        try:
            async with httpx.AsyncClient(base_url=self.base_url) as http:
                resp = await http.get("/view", params=params, timeout=120.0)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise GameaClientError(f"ComfyUI /view unreachable: {exc}") from exc
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        return dest_path


@dataclass
class BridgeClient:
    """Client for bridge/server.py -- the Flask wrapper around Meshroom
    and Blender."""

    base_url: str = "http://127.0.0.1:5050"
    poll_interval_s: float = 2.0
    timeout_s: float = 1800.0

    async def status(self) -> dict:
        try:
            async with httpx.AsyncClient(base_url=self.base_url) as http:
                resp = await http.get("/status", timeout=10.0)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as exc:
            raise GameaClientError(f"bridge /status unreachable: {exc}") from exc

    async def _wait_for_job(self, job_id: str) -> dict:
        deadline = time.monotonic() + self.timeout_s
        async with httpx.AsyncClient(base_url=self.base_url) as http:
            while time.monotonic() < deadline:
                try:
                    resp = await http.get(f"/jobs/{job_id}", timeout=30.0)
                    resp.raise_for_status()
                    job = resp.json()
                except httpx.HTTPError as exc:
                    raise GameaClientError(f"bridge /jobs/{job_id} unreachable: {exc}") from exc
                if job.get("status") in ("done", "error"):
                    return job
                await asyncio.sleep(self.poll_interval_s)
        raise GameaJobError(f"bridge job {job_id} did not finish within {self.timeout_s}s")

    async def run_meshroom(self, images_dir: str) -> dict:
        try:
            async with httpx.AsyncClient(base_url=self.base_url) as http:
                resp = await http.post("/meshroom/run", json={"images_dir": images_dir}, timeout=30.0)
                resp.raise_for_status()
                job_id = resp.json()["job_id"]
        except httpx.HTTPError as exc:
            raise GameaClientError(f"bridge /meshroom/run unreachable: {exc}") from exc
        job = await self._wait_for_job(job_id)
        if job["status"] == "error":
            raise GameaJobError(f"meshroom job failed: {job.get('error')}")
        return job

    async def run_blender_cleanup(self, input_path: str) -> dict:
        try:
            async with httpx.AsyncClient(base_url=self.base_url) as http:
                resp = await http.post("/blender/cleanup", json={"input_path": input_path}, timeout=30.0)
                resp.raise_for_status()
                job_id = resp.json()["job_id"]
        except httpx.HTTPError as exc:
            raise GameaClientError(f"bridge /blender/cleanup unreachable: {exc}") from exc
        job = await self._wait_for_job(job_id)
        if job["status"] == "error":
            raise GameaJobError(f"blender cleanup job failed: {job.get('error')}")
        return job
