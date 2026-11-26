"""Curated static signature and attribute database for safe offline verification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


class StaticAPIData:
    """Represents pre-computed static signature and attribute metadata for a module or object."""

    def __init__(
        self,
        valid_attributes: Set[str],
        function_signatures: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.valid_attributes = valid_attributes
        self.function_signatures = function_signatures or {}


# Static catalog for common standard library and popular packages
STATIC_METADATA_CATALOG: Dict[str, StaticAPIData] = {
    "math": StaticAPIData(
        valid_attributes={
            "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "ceil",
            "comb", "copysign", "cos", "cosh", "degrees", "dist", "e", "erf",
            "erfc", "exp", "exp2", "expm1", "fabs", "factorial", "floor", "fmod",
            "frexp", "fsum", "gamma", "gcd", "hypot", "inf", "isclose", "isfinite",
            "isinf", "isnan", "isqrt", "lcm", "ldexp", "lgamma", "log", "log10",
            "log1p", "log2", "modf", "nan", "nextafter", "perm", "pi", "pow",
            "prod", "radians", "remainder", "sin", "sinh", "sqrt", "tan", "tanh",
            "tau", "trunc", "ulp",
        },
        function_signatures={
            "sqrt": {"pos_only": ["x"], "kw_valid": [], "has_kwargs": False},
            "pow": {"pos_only": ["x", "y"], "kw_valid": [], "has_kwargs": False},
            "gcd": {"pos_only": [], "kw_valid": [], "has_args": True, "has_kwargs": False},
        },
    ),
    "json": StaticAPIData(
        valid_attributes={
            "dump", "dumps", "load", "loads", "JSONDecoder", "JSONDecodeError", "JSONEncoder",
        },
        function_signatures={
            "dumps": {
                "pos_only": [],
                "kw_valid": [
                    "obj", "skipkeys", "ensure_ascii", "check_circular", "allow_nan",
                    "cls", "indent", "separators", "default", "sort_keys",
                ],
                "has_kwargs": False,
            },
            "loads": {
                "pos_only": [],
                "kw_valid": [
                    "s", "cls", "object_hook", "parse_float", "parse_int",
                    "parse_constant", "object_pairs_hook",
                ],
                "has_kwargs": False,
            },
        },
    ),
    "os": StaticAPIData(
        valid_attributes={
            "name", "environ", "getenv", "putenv", "getcwd", "chdir", "listdir",
            "mkdir", "makedirs", "remove", "unlink", "rmdir", "removedirs", "rename",
            "renames", "replace", "stat", "chmod", "walk", "sep", "pathsep", "linesep",
            "path", "system", "urandom", "cpu_count",
        },
        function_signatures={
            "getenv": {"pos_only": [], "kw_valid": ["key", "default"], "has_kwargs": False},
            "remove": {"pos_only": [], "kw_valid": ["path", "dir_fd"], "has_kwargs": False},
        },
    ),
    "requests": StaticAPIData(
        valid_attributes={
            "get", "post", "put", "delete", "head", "options", "patch", "request",
            "Session", "session", "Response", "Request", "exceptions", "codes",
            "auth", "utils", "status_codes",
        },
        function_signatures={
            "get": {
                "pos_only": [],
                "kw_valid": [
                    "url", "params", "data", "headers", "cookies", "files", "auth",
                    "timeout", "allow_redirects", "proxies", "verify", "stream", "cert",
                ],
                "has_kwargs": True,
            },
            "post": {
                "pos_only": [],
                "kw_valid": [
                    "url", "data", "json", "params", "headers", "cookies", "files",
                    "auth", "timeout", "allow_redirects", "proxies", "verify", "stream", "cert",
                ],
                "has_kwargs": True,
            },
        },
    ),
}
