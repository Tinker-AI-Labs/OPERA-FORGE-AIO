"""VIDEA producers -- LLM text and code roles.

An LLM agent is a Producer like any other. There is no separate code path for
it and no separate review loop; the same ``produce -> judge -> revise`` cycle in
``opera.loop`` drives it (spec 3.1).
"""

from __future__ import annotations

from opera.config import RoleConfig
from opera.errors import ProducerError, ProducerUnavailable
from opera.protocols import LLMClient
from opera.schemas import Artifact, Brief


class LLMProducer:
    """Produces text or code by prompting a local model."""

    def __init__(
        self,
        client: LLMClient,
        role: str,
        config: RoleConfig,
        *,
        available: bool = True,
    ) -> None:
        if client is None:
            # Spec 5.4: an absent client is a wiring bug, not a reason to fake.
            raise ProducerError(f"LLMProducer for role {role!r} requires an explicit LLM client")
        self.client = client
        self.name = role
        self.role = role
        self.config = config
        self.kind = config.kind
        self.available = available

    def _prompt(self, brief: Brief) -> str:
        parts: list[str] = []
        if brief.context.strip():
            parts.append(f"Project context:\n{brief.context.strip()}\n")
        parts.append(f"Task: {brief.goal.strip()}")

        if brief.is_revision and brief.prior is not None:
            parts.append(
                "\nYour previous attempt:\n"
                "-----\n"
                f"{brief.prior.content}\n"
                "-----"
            )
            if brief.issues:
                listed = "\n".join(f"- {issue}" for issue in brief.issues)
                parts.append(f"\nA reviewer raised these issues. Address every one:\n{listed}")
            else:
                parts.append("\nThe reviewer was not satisfied but listed no specific issues. "
                             "Improve the weakest part.")
            parts.append("\nReturn the full revised work, not a diff or a summary of changes.")
        return "\n".join(parts)

    async def produce(self, brief: Brief) -> Artifact:
        try:
            content = await self.client.complete(
                system=self.config.system_prompt,
                prompt=self._prompt(brief),
                model=self.config.model,
                timeout=self.config.timeout_s,
                options={"temperature": self.config.temperature, **self.config.options},
                no_think=self.config.no_think,
            )
        except Exception as exc:
            # Transport retries already happened inside the client (spec 5.5).
            # Reaching here means the host is genuinely not answering.
            raise ProducerUnavailable(
                f"role {self.role!r} could not reach its model host: {exc}"
            ) from exc

        if not content.strip():
            raise ProducerError(f"role {self.role!r} returned an empty completion")

        return Artifact(
            kind=self.kind,
            content=content.strip(),
            producer=self.name,
            meta={"model": self.config.model, "role": self.role},
        )
