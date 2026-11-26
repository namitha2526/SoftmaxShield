"""Main verification pipeline orchestrating AST analysis, package verification, API checks, and confidence analysis."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from softmaxshield.analyzer.ast_visitor import parse_source_code
from softmaxshield.analyzer.resolver import ModuleResolver
from softmaxshield.confidence.analyzer import ConfidenceAnalyzer
from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import (
    IssueCategory,
    ModuleType,
    Severity,
    ShieldIssue,
    ShieldReport,
    TokenLogprob,
    VerificationStatus,
)
from softmaxshield.verifier.api_verifier import APIVerifier
from softmaxshield.verifier.package_verifier import PackageVerifier


class ShieldPipeline:
    """End-to-end static verification pipeline."""

    def __init__(self, config: Optional[ShieldConfig] = None) -> None:
        self.config = config or ShieldConfig()
        self.package_verifier = PackageVerifier(self.config)
        self.api_verifier = APIVerifier(self.config)
        self.confidence_analyzer = ConfidenceAnalyzer(self.config)
        self.resolver = ModuleResolver()

    def verify_code(
        self,
        source_code: str,
        file_name: str = "<input>",
        token_logprobs: Optional[List[Union[TokenLogprob, Dict[str, Any]]]] = None,
    ) -> ShieldReport:
        """Execute full verification pipeline on Python source code string."""
        start_time = time.perf_counter()
        issues: List[ShieldIssue] = []

        # 1. AST-Based Code Analysis
        ast_result = parse_source_code(source_code, file_name=file_name)
        if ast_result.syntax_error:
            issues.append(ast_result.syntax_error)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ShieldReport(
                file_name=file_name,
                passed=False,
                total_issues=1,
                issues=issues,
                execution_time_ms=elapsed_ms,
                metadata={"has_syntax_error": True},
            )

        # 2. Package Verification
        verified_pkgs: Dict[str, ModuleType] = {}
        for imp in ast_result.imports:
            top_mod = imp.module_name.split(".")[0] if imp.module_name else ""
            if not top_mod:
                continue

            # Resolve module type
            mod_type = self.resolver.classify_module(top_mod, file_name)
            verified_pkgs[top_mod] = mod_type

            if mod_type == ModuleType.THIRD_PARTY:
                pkg_issue = self.package_verifier.verify_package(
                    import_name=top_mod,
                    file_name=file_name,
                    line=imp.line,
                    col=imp.col,
                )
                if pkg_issue:
                    issues.append(pkg_issue)

            # Check direct imported symbols (e.g. from numpy import fast_average)
            if imp.imported_symbol:
                import_issues = self.api_verifier.verify_import(
                    imp=imp,
                    file_name=file_name,
                    module_type=mod_type,
                )
                issues.extend(import_issues)

        # 3. API & Signature Verification
        for call in ast_result.calls:
            caller_top = call.caller_name.split(".")[0] if call.caller_name else ""
            caller_mod_type = verified_pkgs.get(caller_top, self.resolver.classify_module(caller_top, file_name))

            call_issues = self.api_verifier.verify_call(
                call=call,
                file_name=file_name,
                ast_result=ast_result,
                module_type=caller_mod_type,
            )
            issues.extend(call_issues)

        # 4. Confidence Analysis (Layer gracefully disabled if token_logprobs is None)
        if token_logprobs:
            # Check confidence for package imports
            for imp in ast_result.imports:
                if imp.module_name:
                    conf_issue = self.confidence_analyzer.assess_identifier(
                        identifier=imp.module_name,
                        file_name=file_name,
                        line=imp.line,
                        col=imp.col,
                        token_logprobs=token_logprobs,
                    )
                    if conf_issue:
                        issues.append(conf_issue)

            # Check confidence for calls
            for call in ast_result.calls:
                if call.caller_name:
                    conf_issue = self.confidence_analyzer.assess_identifier(
                        identifier=call.caller_name,
                        file_name=file_name,
                        line=call.line,
                        col=call.col,
                        token_logprobs=token_logprobs,
                    )
                    if conf_issue:
                        issues.append(conf_issue)

        # 5. Formulate Final Report
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if self.config.strict_mode:
            passed = len(issues) == 0
        else:
            # Non-strict mode: passed if no CRITICAL or HIGH issues
            blocking_issues = [i for i in issues if i.severity in (Severity.CRITICAL, Severity.HIGH)]
            passed = len(blocking_issues) == 0

        return ShieldReport(
            file_name=file_name,
            passed=passed,
            total_issues=len(issues),
            issues=issues,
            execution_time_ms=elapsed_ms,
            metadata={
                "imports_count": len(ast_result.imports),
                "calls_count": len(ast_result.calls),
                "strict_mode": self.config.strict_mode,
            },
        )

    def verify_file(
        self,
        file_path: Union[str, Path],
        token_logprobs: Optional[List[Union[TokenLogprob, Dict[str, Any]]]] = None,
    ) -> ShieldReport:
        """Verify code from a local file path."""
        path = Path(file_path)
        if not path.is_file():
            return ShieldReport(
                file_name=str(file_path),
                passed=False,
                total_issues=1,
                issues=[
                    ShieldIssue(
                        file_name=str(file_path),
                        line=1,
                        col=0,
                        detected_identifier=str(file_path),
                        issue_type=IssueCategory.UNVERIFIED_REFERENCE,
                        severity=Severity.CRITICAL,
                        verification_status=VerificationStatus.VERIFIED_INVALID,
                        explanation=f"Target file '{file_path}' does not exist.",
                        suggested_correction=None,
                    )
                ],
                execution_time_ms=0.0,
            )

        source_code = path.read_text(encoding="utf-8", errors="replace")
        return self.verify_code(source_code, file_name=str(path), token_logprobs=token_logprobs)

