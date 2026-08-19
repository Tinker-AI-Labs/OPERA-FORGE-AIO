"""OPERA -- a local, engine-agnostic produce/judge/revise/persist loop.

The core knows nothing about any engine. Engines are data (``EngineSpec``),
not forks of this package.
"""

from .config import LoopConfig, OperaConfig, RoleConfig, load_config
from .errors import (
    JSONParseError,
    LLMTransportError,
    OperaError,
    ProducerError,
    ProducerUnavailable,
)
from .protocols import HumanGate, Judge, LLMClient, Producer
from .schemas import (
    Artifact,
    Bible,
    Brief,
    Ledger,
    Project,
    Run,
    RunStatus,
    Task,
    TaskStatus,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "Artifact", "Bible", "Brief", "Ledger", "Project", "Run", "RunStatus",
    "Task", "TaskStatus", "Verdict",
    "Producer", "Judge", "LLMClient", "HumanGate",
    "OperaConfig", "LoopConfig", "RoleConfig", "load_config",
    "OperaError", "ProducerUnavailable", "ProducerError", "JSONParseError",
    "LLMTransportError",
]
