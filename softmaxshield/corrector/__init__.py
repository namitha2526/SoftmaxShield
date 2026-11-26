"""Corrector package for self-correction feedback loop."""

from softmaxshield.corrector.client import LLMClientProtocol, MockLLMClient
from softmaxshield.corrector.engine import CorrectionResult, SelfCorrectionEngine

__all__ = [
    "LLMClientProtocol",
    "MockLLMClient",
    "SelfCorrectionEngine",
    "CorrectionResult",
]

