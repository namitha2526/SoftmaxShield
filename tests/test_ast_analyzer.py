"""Unit tests for AST analysis, parsing, and symbol resolution."""

import pytest
from softmaxshield.analyzer.ast_visitor import parse_source_code
from softmaxshield.analyzer.resolver import ModuleResolver
from softmaxshield.core.models import IssueCategory, ModuleType, Severity


def test_parse_imports_and_aliases():
    """Verify standard imports and alias extraction."""
    code = """
import os
import numpy as np
import requests.auth as ra
"""
    result = parse_source_code(code, "test.py")
    assert result.syntax_error is None
    assert len(result.imports) == 3

    modules = {i.module_name: i.alias for i in result.imports}
    assert modules["os"] is None
    assert modules["numpy"] == "np"
    assert modules["requests.auth"] == "ra"
    assert result.alias_map["np"] == "numpy"
    assert result.alias_map["ra"] == "requests.auth"


def test_parse_from_imports_and_symbols():
    """Verify from-import extraction with symbols and relative levels."""
    code = """
from math import sqrt as s, pi
from .local_mod import helper
from ..parent_mod import parent_func as pf
"""
    result = parse_source_code(code, "test.py")
    assert result.syntax_error is None
    assert len(result.imports) == 4

    symbols = {(i.module_name, i.imported_symbol): (i.alias, i.is_relative, i.level) for i in result.imports}
    assert (("math", "sqrt")) in symbols
    assert symbols[("math", "sqrt")][0] == "s"

    assert (("local_mod", "helper")) in symbols
    assert symbols[("local_mod", "helper")][1] is True
    assert symbols[("local_mod", "helper")][2] == 1


def test_parse_calls_and_kwargs():
    """Verify function calls, argument count, and keyword arguments."""
    code = """
import json
json.dumps(data, indent=4, sort_keys=True)
my_func(1, 2, *args, **kwargs)
"""
    result = parse_source_code(code, "test.py")
    assert len(result.calls) == 2

    dump_call = result.calls[0]
    assert dump_call.caller_name == "json.dumps"
    assert dump_call.args_count == 1
    assert "indent" in dump_call.kwargs
    assert "sort_keys" in dump_call.kwargs

    generic_call = result.calls[1]
    assert generic_call.caller_name == "my_func"
    assert generic_call.args_count == 3  # 1, 2, *args
    assert generic_call.has_varargs is True
    assert generic_call.has_varkwargs is True


def test_syntax_error_handled_gracefully():
    """Verify that syntax errors produce a structured issue without crashing."""
    bad_code = "def broken_syntax(:\n    pass"
    result = parse_source_code(bad_code, "broken.py")

    assert result.syntax_error is not None
    assert result.syntax_error.issue_type == IssueCategory.INVALID_API
    assert result.syntax_error.severity == Severity.CRITICAL
    assert result.syntax_error.line == 1
    assert "Syntax Error" in result.syntax_error.explanation


def test_module_resolver_classification(tmp_path):
    """Verify classifier correctly distinguishes STDLIB, LOCAL, and THIRD_PARTY."""
    # Create local module file in tmp_path
    local_file = tmp_path / "my_custom_module.py"
    local_file.write_text("def hello(): pass")

    resolver = ModuleResolver(workspace_path=str(tmp_path))

    # Standard library
    assert resolver.classify_module("math") == ModuleType.STDLIB
    assert resolver.classify_module("os.path") == ModuleType.STDLIB
    assert resolver.classify_module("sys") == ModuleType.STDLIB

    # Local module
    assert resolver.classify_module("my_custom_module") == ModuleType.LOCAL

    # Third-party module
    assert resolver.classify_module("requests") == ModuleType.THIRD_PARTY
    assert resolver.classify_module("nonexistent_unknown_pkg") == ModuleType.THIRD_PARTY


def test_resolve_identifier_aliases():
    """Verify symbol alias resolution in call chains."""
    code = """
import numpy as np
from json import dumps as json_dumps
np.zeros((3, 3))
json_dumps({"key": "val"})
"""
    result = parse_source_code(code, "test.py")
    resolver = ModuleResolver()

    assert resolver.resolve_identifier("np.zeros", result) == "numpy.zeros"
    assert resolver.resolve_identifier("json_dumps", result) == "json.dumps"
    assert resolver.resolve_identifier("unrelated_func", result) == "unrelated_func"

