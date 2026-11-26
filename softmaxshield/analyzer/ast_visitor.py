"""AST Visitor and Parser for extracting Python code references."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from softmaxshield.core.models import (
    AttributeReference,
    CallReference,
    ImportReference,
    IssueCategory,
    Severity,
    ShieldIssue,
    VerificationStatus,
)


@dataclass
class ASTAnalysisResult:
    """Aggregated results from AST-based static code analysis."""

    file_name: str
    source_code: str
    syntax_error: Optional[ShieldIssue] = None
    imports: List[ImportReference] = field(default_factory=list)
    calls: List[CallReference] = field(default_factory=list)
    attributes: List[AttributeReference] = field(default_factory=list)
    local_definitions: Set[str] = field(default_factory=set)
    alias_map: Dict[str, str] = field(default_factory=dict)


class ShieldASTVisitor(ast.NodeVisitor):
    """Traverses Python AST to extract imports, calls, attributes, and local symbols."""

    def __init__(self, file_name: str, source_code: str) -> None:
        self.file_name = file_name
        self.source_code = source_code
        self.imports: List[ImportReference] = []
        self.calls: List[CallReference] = []
        self.attributes: List[AttributeReference] = []
        self.local_definitions: Set[str] = set()
        self.alias_map: Dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        """Extract 'import x' and 'import x as y' statements."""
        for alias in node.names:
            line = node.lineno
            col = node.col_offset
            self.imports.append(
                ImportReference(
                    module_name=alias.name,
                    imported_symbol=None,
                    alias=alias.asname,
                    line=line,
                    col=col,
                    end_line=getattr(node, "end_lineno", line),
                    end_col=getattr(node, "end_col_offset", col),
                    is_relative=False,
                    level=0,
                )
            )
            if alias.asname:
                self.alias_map[alias.asname] = alias.name
            else:
                self.alias_map[alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract 'from x import y' and 'from x import y as z' statements."""
        module = node.module or ""
        is_relative = node.level > 0
        line = node.lineno
        col = node.col_offset

        for alias in node.names:
            self.imports.append(
                ImportReference(
                    module_name=module,
                    imported_symbol=alias.name,
                    alias=alias.asname,
                    line=line,
                    col=col,
                    end_line=getattr(node, "end_lineno", line),
                    end_col=getattr(node, "end_col_offset", col),
                    is_relative=is_relative,
                    level=node.level,
                )
            )
            # Map alias or symbol name to fully qualified path
            full_target = f"{module}.{alias.name}" if module else alias.name
            if alias.asname:
                self.alias_map[alias.asname] = full_target
            else:
                self.alias_map[alias.name] = full_target
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record locally defined function names."""
        self.local_definitions.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Record locally defined async function names."""
        self.local_definitions.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record locally defined class names."""
        self.local_definitions.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Record top-level variable names."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_definitions.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Extract function/method calls with arguments and keyword arguments."""
        caller_name = self._resolve_call_name(node.func)
        has_varargs = any(isinstance(arg, ast.Starred) for arg in node.args)
        has_varkwargs = any(kw.arg is None for kw in node.keywords)
        kwargs_list = [kw.arg for kw in node.keywords if kw.arg is not None]

        self.calls.append(
            CallReference(
                caller_name=caller_name,
                args_count=len(node.args),
                kwargs=kwargs_list,
                line=node.lineno,
                col=node.col_offset,
                has_varargs=has_varargs,
                has_varkwargs=has_varkwargs,
            )
        )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Extract attribute access expressions."""
        full_name = self._resolve_call_name(node)
        base_name = self._resolve_call_name(node.value)
        self.attributes.append(
            AttributeReference(
                base=base_name,
                attr=node.attr,
                full_name=full_name,
                line=node.lineno,
                col=node.col_offset,
            )
        )
        self.generic_visit(node)

    def _resolve_call_name(self, node: ast.AST) -> str:
        """Resolve an AST node to a dotted identifier string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_str = self._resolve_call_name(node.value)
            return f"{value_str}.{node.attr}" if value_str else node.attr
        elif isinstance(node, ast.Call):
            return self._resolve_call_name(node.func)
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return ""


def parse_source_code(source_code: str, file_name: str = "<input>") -> ASTAnalysisResult:
    """Parse Python source code into an ASTAnalysisResult, handling syntax errors gracefully.

    Args:
        source_code: The Python code string to parse.
        file_name: Name of the file or stream being parsed.

    Returns:
        ASTAnalysisResult containing extracted references or syntax error issue.
    """
    try:
        tree = ast.parse(source_code, filename=file_name)
    except SyntaxError as err:
        syntax_issue = ShieldIssue(
            file_name=file_name,
            line=err.lineno or 1,
            col=err.offset or 0,
            detected_identifier=err.text.strip() if err.text else "SyntaxError",
            issue_type=IssueCategory.INVALID_API,
            severity=Severity.CRITICAL,
            verification_status=VerificationStatus.VERIFIED_INVALID,
            explanation=f"Syntax Error: {err.msg}",
            suggested_correction=None,
        )
        return ASTAnalysisResult(
            file_name=file_name,
            source_code=source_code,
            syntax_error=syntax_issue,
        )

    visitor = ShieldASTVisitor(file_name=file_name, source_code=source_code)
    visitor.visit(tree)

    return ASTAnalysisResult(
        file_name=file_name,
        source_code=source_code,
        syntax_error=None,
        imports=visitor.imports,
        calls=visitor.calls,
        attributes=visitor.attributes,
        local_definitions=visitor.local_definitions,
        alias_map=visitor.alias_map,
    )

