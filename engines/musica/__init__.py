"""MUSICA -- audio work on the OPERA core."""

from .analysis import AudioStats, AudioUnreadable, analyse
from .gate import AutoApproveGate, CallbackGate, HoldForHuman
from .judge import MusicaJudge, MusicSpec, build_checks, build_judge, coverage_label
from .producers import AceStepProducer, FluidSynthProducer, LLMPlanProducer
from .spec import KINDS, ROUTER_KEYWORDS, build, default_roles

__all__ = [
    "build", "default_roles", "KINDS", "ROUTER_KEYWORDS",
    "LLMPlanProducer", "FluidSynthProducer", "AceStepProducer",
    "MusicaJudge", "MusicSpec", "build_judge", "build_checks", "coverage_label",
    "HoldForHuman", "CallbackGate", "AutoApproveGate",
    "analyse", "AudioStats", "AudioUnreadable",
]
