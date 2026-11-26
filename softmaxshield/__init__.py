"""SoftmaxShield: Multi-layer Static Verification Engine for LLM-Generated Python Code."""

from softmaxshield.analyzer.ast_visitor import ASTAnalysisResult, parse_source_code
from softmaxshield.analyzer.resolver import ModuleResolver
from softmaxshield.confidence.analyzer import ConfidenceAnalyzer, stable_softmax
from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import (
    CallReference,
    CorrectionAttempt,
    ImportReference,
    IssueCategory,
    ModuleType,
    Severity,
    ShieldIssue,
    ShieldReport,
    TokenLogprob,
    VerificationStatus,
)
from softmaxshield.corrector.engine import CorrectionResult, SelfCorrectionEngine
from softmaxshield.pipeline import ShieldPipeline
from softmaxshield.verifier.api_verifier import APIVerifier
from softmaxshield.verifier.package_verifier import PackageVerifier

__version__ = "0.1.0"

__all__ = [
    "ShieldPipeline",
    "ShieldConfig",
    "parse_source_code",
    "ASTAnalysisResult",
    "ModuleResolver",
    "PackageVerifier",
    "APIVerifier",
    "ConfidenceAnalyzer",
    "stable_softmax",
    "SelfCorrectionEngine",
    "CorrectionResult",
    "IssueCategory",
    "Severity",
    "VerificationStatus",
    "ModuleType",
    "ImportReference",
    "CallReference",
    "TokenLogprob",
    "ShieldIssue",
    "ShieldReport",
    "CorrectionAttempt",
]
