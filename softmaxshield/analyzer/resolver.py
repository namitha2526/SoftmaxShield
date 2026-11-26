"""Module and symbol resolver for classifying and tracking imports and references."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Set

from softmaxshield.analyzer.ast_visitor import ASTAnalysisResult
from softmaxshield.core.models import ImportReference, ModuleType


class ModuleResolver:
    """Classifies imports into STDLIB, LOCAL, or THIRD_PARTY, and resolves symbol aliases."""

    def __init__(self, workspace_path: Optional[str] = None) -> None:
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()
        # Python 3.10+ provides sys.stdlib_module_names
        self.stdlib_names: Set[str] = set(getattr(sys, "stdlib_module_names", set()))
        if not self.stdlib_names:
            # Fallback list of major standard library modules
            self.stdlib_names = {
                "abc", "argparse", "array", "ast", "asyncio", "base64", "binascii",
                "bisect", "builtins", "bz2", "calendar", "cmath", "collections",
                "concurrent", "configparser", "contextlib", "copy", "csv", "ctypes",
                "dataclasses", "datetime", "decimal", "difflib", "dis", "doctest",
                "email", "enum", "errno", "faulthandler", "fcntl", "filecmp",
                "fileinput", "fnmatch", "fractions", "functools", "gc", "getopt",
                "getpass", "gettext", "glob", "gzip", "hashlib", "heapq", "hmac",
                "html", "http", "idlelib", "imaplib", "imghdr", "importlib",
                "inspect", "io", "ipaddress", "itertools", "json", "keyword",
                "linecache", "locale", "logging", "lzma", "mailbox", "mailcap",
                "marshal", "math", "mimetypes", "mmap", "modulefinder", "multiprocessing",
                "netrc", "nntplib", "numbers", "operator", "optparse", "os", "pathlib",
                "pdb", "pickle", "pickletools", "pkgutil", "platform", "plistlib",
                "poplib", "posix", "pprint", "profile", "pstats", "pty", "pwd",
                "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random", "re",
                "readline", "reprlib", "resource", "rlcompleter", "runpy", "sched",
                "secrets", "select", "selectors", "shelve", "shlex", "shutil", "signal",
                "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver",
                "sqlite3", "ssl", "stat", "statistics", "string", "stringprep",
                "struct", "subprocess", "sunau", "symbol", "symtable", "sys",
                "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile",
                "termios", "test", "textwrap", "threading", "time", "timeit",
                "tkinter", "token", "tokenize", "trace", "traceback", "tracemalloc",
                "tty", "turtle", "turtledemo", "types", "typing", "unicodedata",
                "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave",
                "weakref", "webbrowser", "wsgiref", "xdrlib", "xml", "xmlrpc",
                "zipapp", "zipfile", "zipimport", "zlib", "_thread"
            }

    def classify_module(self, module_name: str, file_path: Optional[str] = None) -> ModuleType:
        """Classify a module name as STDLIB, LOCAL, or THIRD_PARTY.

        Args:
            module_name: Dotted or top-level module name (e.g., 'math', 'requests.auth').
            file_path: Path of the file containing the import.

        Returns:
            ModuleType enum value.
        """
        if not module_name:
            return ModuleType.UNKNOWN

        top_level = module_name.split(".")[0]

        # 1. Check Python Standard Library
        if top_level in self.stdlib_names:
            return ModuleType.STDLIB

        # 2. Check if it is a local module in file directory or workspace
        if file_path:
            file_dir = Path(file_path).parent
            if self._is_local_path(file_dir, top_level):
                return ModuleType.LOCAL

        if self._is_local_path(self.workspace_path, top_level):
            return ModuleType.LOCAL

        # 3. Otherwise classified as third-party
        return ModuleType.THIRD_PARTY

    def _is_local_path(self, base_dir: Path, module_name: str) -> bool:
        """Check if module_name exists as a python file or package directory in base_dir."""
        py_file = base_dir / f"{module_name}.py"
        pkg_dir = base_dir / module_name
        pkg_init = pkg_dir / "__init__.py"
        return py_file.is_file() or (pkg_dir.is_dir() and pkg_init.is_file())

    def resolve_identifier(self, identifier: str, ast_result: ASTAnalysisResult) -> str:
        """Resolve an identifier or call name using alias mapping and imports.

        For example:
            - 'np.zeros' with alias 'np' -> 'numpy' becomes 'numpy.zeros'
            - 'loads' with alias 'loads' -> 'json.loads' becomes 'json.loads'
        """
        if not identifier:
            return ""

        parts = identifier.split(".", 1)
        root = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""

        if root in ast_result.alias_map:
            resolved_root = ast_result.alias_map[root]
            if remainder:
                return f"{resolved_root}.{remainder}"
            return resolved_root

        return identifier

