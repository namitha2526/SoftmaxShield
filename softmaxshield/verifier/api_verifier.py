"""API, attribute, and signature verification engine."""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from softmaxshield.analyzer.ast_visitor import ASTAnalysisResult
from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import (
    CallReference,
    IssueCategory,
    ModuleType,
    Severity,
    ShieldIssue,
    VerificationStatus,
)
from softmaxshield.verifier.metadata_db import STATIC_METADATA_CATALOG
from softmaxshield.verifier.package_verifier import levenshtein_distance

logger = logging.getLogger(__name__)


# Generic container nouns that should not be matched as substring targets for operational verbs
GENERIC_CONTAINER_NOUNS: Set[str] = {
    "array", "data", "matrix", "tensor", "vector", "list", "dict",
    "series", "object", "item", "items", "val", "value", "values", "type"
}


class APIVerifier:
    """Verifies functions, classes, attributes, and keyword arguments.

    Security Guarantee: Never executes or imports untrusted generated code.
    Only inspects trusted standard-library modules or safely available installed packages.
    """

    def __init__(self, config: Optional[ShieldConfig] = None) -> None:
        self.config = config or ShieldConfig()
        self._trusted_modules_cache: Dict[str, Any] = {}

    def _get_trusted_module(self, module_name: str) -> Optional[Any]:
        """Safely import trusted standard library or installed module for inspection."""
        if not module_name:
            return None

        if module_name in self._trusted_modules_cache:
            return self._trusted_modules_cache[module_name]

        # 1. Try direct import of full module path
        try:
            mod = importlib.import_module(module_name)
            self._trusted_modules_cache[module_name] = mod
            return mod
        except (ImportError, AttributeError, ValueError, Exception):
            pass

        # 2. If sub-path, import top-level module and traverse attributes
        if "." in module_name:
            parts = module_name.split(".")
            try:
                curr = importlib.import_module(parts[0])
                for part in parts[1:]:
                    if hasattr(curr, part):
                        curr = getattr(curr, part)
                    else:
                        return None
                self._trusted_modules_cache[module_name] = curr
                return curr
            except Exception as exc:
                logger.debug("Could not traverse '%s' for inspection: %s", module_name, exc)
                return None

        return None

    def _find_closest_attribute(self, target_name: str, available_attrs: Set[str]) -> Optional[str]:
        """Find the most similar valid attribute name using Levenshtein distance and substring matching."""
        target_lower = target_name.lower()
        best_name: Optional[str] = None
        best_score = -1.0

        for attr in available_attrs:
            if attr.startswith("_"):
                continue
            attr_lower = attr.lower()

            # Guard against false semantic matches where generic container noun is a suffix
            # e.g., 'normalize_array' should never match 'array'
            if attr_lower in GENERIC_CONTAINER_NOUNS and attr_lower != target_lower:
                continue

            # Substring match (e.g. 'sqrt' in 'fast_sqrt', 'average' in 'fast_average')
            if attr_lower in target_lower or target_lower in attr_lower:
                # Guard: Short / single-character attributes (e.g. 'e', 'pi', 'c') must not match arbitrary long words
                if len(attr_lower) < 3 and len(target_lower) > 3:
                    continue
                if len(attr_lower) / max(len(target_lower), 1) < 0.25 and not (f"_{attr_lower}" in target_lower or f"{attr_lower}_" in target_lower):
                    continue

                score = 0.85 + (len(attr_lower) / max(len(target_lower), len(attr_lower)) * 0.14)
                if score > best_score:
                    best_score = score
                    best_name = attr
                continue

            dist = levenshtein_distance(target_lower, attr_lower)
            max_len = max(len(target_lower), len(attr_lower))
            sim = 1.0 - (dist / max_len) if max_len > 0 else 0.0

            if dist <= 3 and sim >= 0.5 and sim > best_score:
                best_score = sim
                best_name = attr

        return best_name

    def verify_import(
        self,
        imp: Any,
        file_name: str,
        module_type: ModuleType,
    ) -> List[ShieldIssue]:
        """Verify that an imported symbol exists on an imported module."""
        issues: List[ShieldIssue] = []
        if not getattr(imp, "imported_symbol", None) or not getattr(imp, "module_name", None):
            return issues

        trusted_mod = self._get_trusted_module(imp.module_name)
        if trusted_mod is None:
            return issues

        symbol_name = imp.imported_symbol
        try:
            has_symbol = hasattr(trusted_mod, symbol_name)
        except Exception:
            return issues

        if not has_symbol:
            try:
                available = set(dir(trusted_mod))
            except Exception:
                available = set()
            closest = self._find_closest_attribute(symbol_name, available)
            full_ident = f"{imp.module_name}.{symbol_name}"
            issues.append(
                ShieldIssue(
                    file_name=file_name,
                    line=imp.line,
                    col=imp.col,
                    detected_identifier=full_ident,
                    issue_type=IssueCategory.INVALID_API,
                    severity=Severity.HIGH,
                    verification_status=VerificationStatus.VERIFIED_INVALID,
                    explanation=f"Symbol '{symbol_name}' does not exist on imported module '{imp.module_name}'.",
                    suggested_correction=f"{imp.module_name}.{closest}" if closest else None,
                )
            )
        return issues

    def verify_call(
        self,
        call: CallReference,
        file_name: str,
        ast_result: ASTAnalysisResult,
        module_type: ModuleType,
    ) -> List[ShieldIssue]:
        """Verify a function or method call against inspected or static signatures."""
        issues: List[ShieldIssue] = []
        caller_name = call.caller_name
        if not caller_name:
            return issues

        # Resolve aliases (e.g., 'np.zeros' -> 'numpy.zeros', 'loads' -> 'json.loads')
        resolved_name = self._resolve_name(caller_name, ast_result)

        # Check if caller is locally defined in the file
        top_symbol = resolved_name.split(".")[0]
        if top_symbol in ast_result.local_definitions:
            return issues

        # Attempt verification via inspected module or static database
        if "." in resolved_name:
            module_path, func_name = resolved_name.rsplit(".", 1)
        else:
            module_path, func_name = "", resolved_name

        # Case 1: Check static metadata database first
        if module_path in STATIC_METADATA_CATALOG:
            static_mod = STATIC_METADATA_CATALOG[module_path]
            if func_name not in static_mod.valid_attributes:
                closest = self._find_closest_attribute(func_name, static_mod.valid_attributes)
                issues.append(
                    ShieldIssue(
                        file_name=file_name,
                        line=call.line,
                        col=call.col,
                        detected_identifier=resolved_name,
                        issue_type=IssueCategory.INVALID_API,
                        severity=Severity.HIGH,
                        verification_status=VerificationStatus.VERIFIED_INVALID,
                        explanation=(
                            f"API function or attribute '{func_name}' does not exist on module '{module_path}'."
                        ),
                        suggested_correction=f"{module_path}.{closest}" if closest else None,
                    )
                )
                return issues

            # Verify kwargs against static signature if available
            sig_info = static_mod.function_signatures.get(func_name)
            if sig_info:
                pos_only = set(sig_info.get("pos_only", []))
                kw_valid = set(sig_info.get("kw_valid", []))
                has_kwargs = sig_info.get("has_kwargs", False)

                for kw in call.kwargs:
                    if kw in pos_only:
                        issues.append(
                            ShieldIssue(
                                file_name=file_name,
                                line=call.line,
                                col=call.col,
                                detected_identifier=f"{resolved_name}({kw}=...)",
                                issue_type=IssueCategory.INVALID_ARGUMENT,
                                severity=Severity.HIGH,
                                verification_status=VerificationStatus.VERIFIED_INVALID,
                                explanation=(
                                    f"Argument '{kw}' in '{resolved_name}' is positional-only and cannot be passed as a keyword argument."
                                ),
                                suggested_correction=None,
                            )
                        )
                    elif not has_kwargs and kw not in kw_valid:
                        closest_kw = self._find_closest_attribute(kw, kw_valid)
                        issues.append(
                            ShieldIssue(
                                file_name=file_name,
                                line=call.line,
                                col=call.col,
                                detected_identifier=f"{resolved_name}({kw}=...)",
                                issue_type=IssueCategory.INVALID_ARGUMENT,
                                severity=Severity.HIGH,
                                verification_status=VerificationStatus.VERIFIED_INVALID,
                                explanation=(
                                    f"Unexpected keyword argument '{kw}' for '{resolved_name}'. Valid arguments are: {sorted(list(kw_valid))}."
                                ),
                                suggested_correction=closest_kw,
                            )
                        )
                return issues

        # Case 2: Inspect module safely if available
        trusted_mod = self._get_trusted_module(module_path)
        if trusted_mod is not None:
            # If trusted_mod is a function or method (not module/class), func_name is a chained method call
            # on the returned instance (e.g. select().where()), which static inspection cannot invalidate.
            if callable(trusted_mod) and not inspect.isclass(trusted_mod):
                return issues

            try:
                has_attr = hasattr(trusted_mod, func_name)
            except Exception:
                # Dynamic proxies (e.g., flask.request outside request context) raise on attribute check
                return issues

            if not has_attr:
                try:
                    available = set(dir(trusted_mod))
                except Exception:
                    available = set()
                closest = self._find_closest_attribute(func_name, available)
                issues.append(
                    ShieldIssue(
                        file_name=file_name,
                        line=call.line,
                        col=call.col,
                        detected_identifier=resolved_name,
                        issue_type=IssueCategory.INVALID_API,
                        severity=Severity.HIGH,
                        verification_status=VerificationStatus.VERIFIED_INVALID,
                        explanation=f"Attribute '{func_name}' does not exist on inspected module '{module_path}'.",
                        suggested_correction=f"{module_path}.{closest}" if closest else None,
                    )
                )
                return issues

            try:
                target_callable = getattr(trusted_mod, func_name)
            except Exception:
                return issues
            if callable(target_callable):
                try:
                    # Unwrap decorators to inspect genuine signature
                    unwrapped = inspect.unwrap(target_callable)
                    sig = inspect.signature(unwrapped)
                    kw_issues = self._verify_signature_kwargs(
                        resolved_name, sig, call.kwargs, file_name, call.line, call.col
                    )
                    issues.extend(kw_issues)
                except (ValueError, TypeError) as sig_err:
                    logger.debug("Signature inspection inconclusive for %s: %s", resolved_name, sig_err)
                    # Report inconclusive rather than falsely declaring invalid
                    if self.config.strict_mode:
                        issues.append(
                            ShieldIssue(
                                file_name=file_name,
                                line=call.line,
                                col=call.col,
                                detected_identifier=resolved_name,
                                issue_type=IssueCategory.UNVERIFIED_REFERENCE,
                                severity=Severity.INFO,
                                verification_status=VerificationStatus.INCONCLUSIVE,
                                explanation=f"Signature for '{resolved_name}' is dynamic or C-accelerated: {sig_err}",
                                suggested_correction=None,
                            )
                        )
            return issues

        # Case 3: Module is uninspectable (third-party not installed, dynamic reference)
        # Report INCONCLUSIVE only in strict mode, or as UNVERIFIED_REFERENCE info
        if module_type == ModuleType.THIRD_PARTY and not self.config.offline_mode:
            issues.append(
                ShieldIssue(
                    file_name=file_name,
                    line=call.line,
                    col=call.col,
                    detected_identifier=resolved_name,
                    issue_type=IssueCategory.UNVERIFIED_REFERENCE,
                    severity=Severity.INFO if not self.config.strict_mode else Severity.WARNING,
                    verification_status=VerificationStatus.INCONCLUSIVE,
                    explanation=(
                        f"Target module '{module_path}' is not installed locally; static API inspection is inconclusive."
                    ),
                    suggested_correction=None,
                )
            )

        return issues

    def _verify_signature_kwargs(
        self,
        func_name: str,
        sig: inspect.Signature,
        passed_kwargs: List[str],
        file_name: str,
        line: int,
        col: int,
    ) -> List[ShieldIssue]:
        """Check provided keyword arguments against an inspected inspect.Signature object."""
        issues: List[ShieldIssue] = []
        pos_only: Set[str] = set()
        valid_kwargs: Set[str] = set()
        has_varkw = False

        for param_name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                pos_only.add(param_name)
            elif param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                valid_kwargs.add(param_name)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                has_varkw = True

        # If function accepts **kwargs, arbitrary kwargs are allowed
        if has_varkw:
            # Check for positional-only argument violations even if **kwargs exists
            for kw in passed_kwargs:
                if kw in pos_only:
                    issues.append(
                        ShieldIssue(
                            file_name=file_name,
                            line=line,
                            col=col,
                            detected_identifier=f"{func_name}({kw}=...)",
                            issue_type=IssueCategory.INVALID_ARGUMENT,
                            severity=Severity.HIGH,
                            verification_status=VerificationStatus.VERIFIED_INVALID,
                            explanation=(
                                f"Argument '{kw}' in '{func_name}' is positional-only and cannot be passed by keyword."
                            ),
                            suggested_correction=None,
                        )
                    )
            return issues

        for kw in passed_kwargs:
            if kw in pos_only:
                issues.append(
                    ShieldIssue(
                        file_name=file_name,
                        line=line,
                        col=col,
                        detected_identifier=f"{func_name}({kw}=...)",
                        issue_type=IssueCategory.INVALID_ARGUMENT,
                        severity=Severity.HIGH,
                        verification_status=VerificationStatus.VERIFIED_INVALID,
                        explanation=(
                            f"Argument '{kw}' in '{func_name}' is positional-only and cannot be passed by keyword."
                        ),
                        suggested_correction=None,
                    )
                )
            elif kw not in valid_kwargs:
                closest = self._find_closest_attribute(kw, valid_kwargs)
                issues.append(
                    ShieldIssue(
                        file_name=file_name,
                        line=line,
                        col=col,
                        detected_identifier=f"{func_name}({kw}=...)",
                        issue_type=IssueCategory.INVALID_ARGUMENT,
                        severity=Severity.HIGH,
                        verification_status=VerificationStatus.VERIFIED_INVALID,
                        explanation=(
                            f"Invalid keyword argument '{kw}' for '{func_name}'. "
                            f"Supported arguments: {sorted(list(valid_kwargs))}."
                        ),
                        suggested_correction=closest,
                    )
                )

        return issues

    def _resolve_name(self, name: str, ast_result: ASTAnalysisResult) -> str:
        """Resolve aliases in an identifier."""
        if not name:
            return ""
        parts = name.split(".", 1)
        root = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        if root in ast_result.alias_map:
            resolved_root = ast_result.alias_map[root]
            return f"{resolved_root}.{rest}" if rest else resolved_root
        return name
