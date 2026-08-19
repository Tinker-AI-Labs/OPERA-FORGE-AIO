"""VIDEA's judges: an LLM reviewer for text/code, frame sampling for video."""

from __future__ import annotations

from opera.judges import FrameSampleJudge, LLMJudge, VisionJudge
from opera.protocols import Judge, LLMClient
from opera.schemas import Artifact, Task, Verdict

VIDEA_JUDGE_SYSTEM = (
    "You review work for a film production. Assess it against the task goal and "
    "the project context -- continuity with established characters, facts and "
    "style matters as much as craft.\n"
    "Reply with JSON only:\n"
    '{"score": 0.0-1.0, "passed": true|false, "issues": ["..."]}\n'
    "Issues must be specific and actionable. Do not restate the work."
)


class VideaJudge:
    """Dispatches on artifact kind, so one engine judge covers all its kinds.

    Each delegate reports its own ``judged`` value, and this wrapper passes it
    through untouched -- a frame-sampled video verdict stays ``judged="frames"``
    rather than being laundered into ``"artifact"``.
    """

    name = "videa-judge"

    def __init__(self, text_judge: Judge, video_judge: Judge | None = None) -> None:
        self.text_judge = text_judge
        self.video_judge = video_judge

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict:
        if artifact.kind == "video" and self.video_judge is not None:
            return await self.video_judge.evaluate(task, artifact, context)
        return await self.text_judge.evaluate(task, artifact, context)


def build_judge(
    client: LLMClient,
    *,
    model: str,
    vision_model: str,
    threshold: float = 0.7,
    frames: int = 4,
    with_video: bool = True,
) -> VideaJudge:
    text = LLMJudge(client, model=model, name="videa-review", threshold=threshold,
                    system=VIDEA_JUDGE_SYSTEM)
    video = None
    if with_video:
        video = FrameSampleJudge(
            VisionJudge(client, model=vision_model, name="videa-vision", threshold=threshold),
            frames=frames, threshold=threshold, name="videa-frames",
        )
    return VideaJudge(text, video)
