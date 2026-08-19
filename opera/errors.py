"""Error taxonomy for OPERA.

Every failure mode the loop can distinguish gets its own type. The loop cares
about the difference between "the transport blipped" (retry, does not consume a
revision attempt -- spec 5.5) and "the model produced something unusable"
(structured parse error, spec 5.3).
"""

from __future__ import annotations


class OperaError(Exception):
    """Base for everything raised by the core."""


class ConfigError(OperaError):
    """Missing or contradictory configuration."""


class LLMError(OperaError):
    """Base for LLM client failures."""


class LLMTransportError(LLMError):
    """Network/connection/timeout failure talking to the model host.

    Retryable. Retries are bounded and MUST NOT consume a revision attempt.
    """

    def __init__(self, message: str, *, attempts: int = 0, cause: Exception | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.cause = cause


class LLMResponseError(LLMError):
    """The host answered, but with a non-success status or empty body."""

    def __init__(self, message: str, *, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class JSONParseError(LLMError):
    """Model output could not be parsed as the expected JSON.

    Structured on purpose: spec 5.3 forbids returning ``None`` and swallowing
    the evidence. The raw text is carried so callers can log what actually came
    back instead of guessing at "model quality".
    """

    def __init__(self, message: str, *, raw: str = "", stage: str = ""):
        super().__init__(message)
        self.raw = raw
        self.stage = stage

    def __str__(self) -> str:  # pragma: no cover - formatting only
        head = super().__str__()
        if self.stage:
            head = f"[{self.stage}] {head}"
        excerpt = self.raw[:400].replace("\n", "\\n")
        return f"{head} | raw={excerpt!r}"


class ProducerUnavailable(OperaError):
    """A producer's backend is not reachable.

    Never a reason to synthesise output. The task is deferred with this reason.
    """


class ProducerError(OperaError):
    """A producer was available but failed while producing."""


class JudgeError(OperaError):
    """A judge could not render a verdict."""


class PlannerError(OperaError):
    """The planner produced nothing usable at all (not merely invalid fields --
    invalid fields are corrected, see spec 4.5)."""


class RegistryError(OperaError):
    """Unknown or duplicate engine registration."""


class ProjectStoreError(OperaError):
    """Project could not be read from or written to disk."""
