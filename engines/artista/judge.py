"""ARTISTA's judge: a real vision model over the produced image."""

from __future__ import annotations

from opera.judges import LLMJudge, VisionJudge
from opera.protocols import Judge, LLMClient
from opera.schemas import Artifact, Task, Verdict

ARTISTA_VISION_SYSTEM = (
    "You review generated images. Assess the image against the brief and the "
    "project's established style, palette and subject continuity.\n"
    "Reply with JSON only:\n"
    '{"score": 0.0-1.0, "passed": true|false, "issues": ["..."]}\n'
    "Issues must describe what to change in the next generation, specifically. "
    "Judge only what you can see."
)

ARTISTA_PROMPT_SYSTEM = (
    "You review image-generation prompts. Assess whether the prompt would "
    "plausibly produce what the brief asks for, in the project's established style.\n"
    "Reply with JSON only:\n"
    '{"score": 0.0-1.0, "passed": true|false, "issues": ["..."]}'
)


class ArtistaJudge:
    """Routes on artifact kind and preserves each delegate's ``judged`` value.

    A prompt is reviewed as a plan, and says so. Only the vision pass over real
    pixels reports ``judged="artifact"``.
    """

    name = "artista-judge"

    def __init__(self, image_judge: Judge, prompt_judge: Judge) -> None:
        self.image_judge = image_judge
        self.prompt_judge = prompt_judge

    async def evaluate(self, task: Task, artifact: Artifact, context: str) -> Verdict:
        if artifact.kind == "image":
            return await self.image_judge.evaluate(task, artifact, context)
        return await self.prompt_judge.evaluate(task, artifact, context)


def build_judge(client: LLMClient, *, vision_model: str, text_model: str,
                threshold: float = 0.7) -> ArtistaJudge:
    return ArtistaJudge(
        image_judge=VisionJudge(client, model=vision_model, name="artista-vision",
                                threshold=threshold, system=ARTISTA_VISION_SYSTEM,
                                judged="artifact"),
        # A prompt review is a plan review. Saying `judged="plan"` is the whole
        # point of that field (spec 3.2).
        prompt_judge=LLMJudge(client, model=text_model, name="artista-prompt-review",
                              threshold=threshold, system=ARTISTA_PROMPT_SYSTEM,
                              judged="plan"),
    )
