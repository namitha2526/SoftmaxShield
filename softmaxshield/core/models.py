"""Core domain models for SoftmaxShield verification engine."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class IssueCategory(str, enum.Enum):
    """Categorization of detected code verification issues."""

    INVALID_PACKAGE = "INVALID_PACKAGE"
    SUSPICIOUS_PACKAGE = "SUSPICIOUS_PACKAGE"
    INVALID_IMPORT = "INVALID_IMPORT"
    INVALID_API = "INVALID_API"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNVERIFIED_REFERENCE = "UNVERIFIED_REFERENCE"


class Severity(str, enum.Enum):
    """Severity levels for detected issues."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WARNING = "WARNING"
    INFO = "INFO"


class VerificationStatus(str, enum.Enum):
    """Status resulting from static verification checks."""

    VERIFIED_VALID = "VERIFIED_VALID"
    VERIFIED_INVALID = "VERIFIED_INVALID"
    SUSPICIOUS = "SUSPICIOUS"
    INCONCLUSIVE = "INCONCLUSIVE"
    SKIPPED = "SKIPPED"


class ModuleType(str, enum.Enum):
    """Classification of Python modules."""

    STDLIB = "STDLIB"
    THIRD_PARTY = "THIRD_PARTY"
    LOCAL = "LOCAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class ImportReference:
    """Represents a detected import statement in code."""

    module_name: str
    imported_symbol: Optional[str] = None
    alias: Optional[str] = None
    line: int = 1
    col: int = 0
    end_line: Optional[int] = None
    end_col: Optional[int] = None
    is_relative: bool = False
    level: int = 0


@dataclass
class CallReference:
    """Represents a detected function or method call."""

    caller_name: str
    args_count: int = 0
    kwargs: List[str] = field(default_factory=list)
    line: int = 1
    col: int = 0
    has_varargs: bool = False
    has_varkwargs: bool = False


@dataclass
class AttributeReference:
    """Represents an attribute access in code."""

    base: str
    attr: str
    full_name: str
    line: int = 1
    col: int = 0


@dataclass
class TokenLogprob:
    """Token-level log probability data from an LLM."""

    token: str
    logprob: float
    top_logprobs: Optional[List[Dict[str, float]]] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None


@dataclass
class ShieldIssue:
    """A detected issue found in LLM-generated code."""

    file_name: str
    line: int
    col: int
    detected_identifier: str
    issue_type: IssueCategory
    severity: Severity
    verification_status: VerificationStatus
    explanation: str
    suggested_correction: Optional[str] = None
    confidence_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the issue into a serializable dictionary."""
        return {
            "file_name": self.file_name,
            "line": self.line,
            "col": self.col,
            "detected_identifier": self.detected_identifier,
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "verification_status": self.verification_status.value,
            "explanation": self.explanation,
            "suggested_correction": self.suggested_correction,
            "confidence_score": round(self.confidence_score, 4) if self.confidence_score is not None else None,
        }


@dataclass
class ShieldReport:
    """Complete verification report for a file or code snippet."""

    file_name: str
    passed: bool
    total_issues: int
    issues: List[ShieldIssue] = field(default_factory=list)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report into a serializable dictionary matching the required schema."""
        return {
            "file_name": self.file_name,
            "passed": self.passed,
            "total_issues": self.total_issues,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": self.metadata,
        }


@dataclass
class CorrectionAttempt:
    """Records one round of self-correction."""

    round_number: int
    code_before: str
    code_after: str
    issues_found: List[ShieldIssue]
    resolved: bool
    diff: str = ""

