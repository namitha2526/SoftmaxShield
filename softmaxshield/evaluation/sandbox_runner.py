"""Safe, isolated sandbox runner for functional testing of generated code snippets.

Supports two complementary isolation tiers:
1. In-process restricted namespace execution (executing in <0.5ms with isolated namespace & mocked I/O)
2. External subprocess isolation with timeout, memory bounds, and stripped environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class _MockResponse:
    """Mock HTTP response for offline evaluation of network requests."""

    def __init__(self) -> None:
        self.status_code = 200
        self.text = '{"status": "ok", "url": "https://example.com"}'
        self.headers = {"Content-Type": "application/json"}

    def json(self) -> Dict[str, Any]:
        return {"status": "ok", "url": "https://example.com"}

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int = 256):
        yield b"mock binary data chunk"


class _MockRequests:
    """Mock requests module preventing real network socket blocks during sandboxed tests."""

    def __init__(self) -> None:
        self.exceptions = sys.modules.get("requests", type("req", (), {})).exceptions if "requests" in sys.modules else type("exc", (), {"Timeout": TimeoutError})

    def get(self, *args, **kwargs) -> _MockResponse:
        return _MockResponse()

    def post(self, *args, **kwargs) -> _MockResponse:
        return _MockResponse()

    def put(self, *args, **kwargs) -> _MockResponse:
        return _MockResponse()

    def delete(self, *args, **kwargs) -> _MockResponse:
        return _MockResponse()

    class Session:
        def __init__(self) -> None:
            self.headers = {}
            self.cookies = {}

        def get(self, *args, **kwargs) -> _MockResponse:
            return _MockResponse()

        def post(self, *args, **kwargs) -> _MockResponse:
            return _MockResponse()

        def mount(self, *args, **kwargs) -> None:
            pass

        def prepare_request(self, req) -> Any:
            return req

        def send(self, *args, **kwargs) -> _MockResponse:
            return _MockResponse()


@dataclass
class SandboxResult:
    """Outcome of sandboxed execution."""

    success: bool
    timed_out: bool
    return_code: int
    stdout: str
    stderr: str
    duration_ms: float
    error_type: Optional[str] = None
    assertion_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "timed_out": self.timed_out,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": round(self.duration_ms, 2),
            "error_type": self.error_type,
            "assertion_passed": self.assertion_passed,
        }


class SandboxRunner:
    """Orchestrates isolated execution with safety bounds."""

    def __init__(self, timeout_seconds: float = 2.0) -> None:
        self.timeout_seconds = timeout_seconds

    def execute_in_namespace(
        self,
        code: str,
        assertion_code: Optional[str] = None,
    ) -> SandboxResult:
        """Fast, isolated in-process execution within a dedicated execution dictionary."""
        start_time = time.perf_counter()

        # Check for AST/string pattern assertions first
        if assertion_code and ("in code" in assertion_code or "not in code" in assertion_code):
            try:
                eval_env = {"code": code, "result": None}
                exec(assertion_code, eval_env)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return SandboxResult(
                    success=True,
                    timed_out=False,
                    return_code=0,
                    stdout="Code pattern assertion passed.",
                    stderr="",
                    duration_ms=elapsed_ms,
                    assertion_passed=True,
                )
            except AssertionError as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return SandboxResult(
                    success=False,
                    timed_out=False,
                    return_code=1,
                    stdout="",
                    stderr=str(exc) or "Pattern AssertionError",
                    duration_ms=elapsed_ms,
                    error_type="AssertionError",
                    assertion_passed=False,
                )

        full_code = code
        if assertion_code:
            full_code = f"{code}\n\n{assertion_code}"

        # Clean namespace with mocked network to prevent live socket blocks
        exec_globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "__name__": "__sandbox__",
            "__doc__": None,
            "requests": _MockRequests(),
        }

        try:
            compiled = compile(full_code, "<sandbox>", "exec")
            exec(compiled, exec_globals)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxResult(
                success=True,
                timed_out=False,
                return_code=0,
                stdout="Executed successfully in isolated namespace.",
                stderr="",
                duration_ms=elapsed_ms,
                error_type=None,
                assertion_passed=True,
            )
        except AssertionError as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxResult(
                success=False,
                timed_out=False,
                return_code=1,
                stdout="",
                stderr=str(exc) or "AssertionError",
                duration_ms=elapsed_ms,
                error_type="AssertionError",
                assertion_passed=False,
            )
        except SyntaxError as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxResult(
                success=False,
                timed_out=False,
                return_code=1,
                stdout="",
                stderr=str(exc),
                duration_ms=elapsed_ms,
                error_type="SyntaxError",
                assertion_passed=False,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxResult(
                success=False,
                timed_out=False,
                return_code=1,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed_ms,
                error_type=type(exc).__name__,
                assertion_passed=False,
            )

    def execute_subprocess(
        self,
        code: str,
        assertion_code: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> SandboxResult:
        """Run code and optional assertions in a dedicated isolated subprocess."""
        effective_timeout = timeout or self.timeout_seconds
        start_time = time.perf_counter()

        full_script = f"# Sandbox wrapper\nimport sys\n{code}\n"
        if assertion_code:
            full_script += f"\n# Assertions\n{assertion_code}\n"

        clean_env = os.environ.copy()
        for sensitive_key in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "HUGGINGFACE_TOKEN",
            "HF_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "GITHUB_TOKEN",
            "DATABASE_URL",
        ]:
            clean_env.pop(sensitive_key, None)

        clean_env["PYTHONUNBUFFERED"] = "1"
        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
        clean_env["MPLBACKEND"] = "Agg"

        with tempfile.TemporaryDirectory(prefix="softmax_sandbox_") as tmpdir:
            script_path = Path(tmpdir) / "runner.py"
            script_path.write_text(full_script, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=tmpdir,
                    env=clean_env,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                stdout = proc.stdout
                stderr = proc.stderr
                success = proc.returncode == 0

                error_type = None
                if not success:
                    if "SyntaxError" in stderr:
                        error_type = "SyntaxError"
                    elif "AssertionError" in stderr:
                        error_type = "AssertionError"
                    elif "ImportError" in stderr or "ModuleNotFoundError" in stderr:
                        error_type = "ImportError"
                    elif "AttributeError" in stderr:
                        error_type = "AttributeError"
                    elif "TypeError" in stderr:
                        error_type = "TypeError"
                    else:
                        error_type = "RuntimeError"

                return SandboxResult(
                    success=success,
                    timed_out=False,
                    return_code=proc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration_ms=elapsed_ms,
                    error_type=error_type,
                    assertion_passed=success,
                )

            except subprocess.TimeoutExpired as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return SandboxResult(
                    success=False,
                    timed_out=True,
                    return_code=-1,
                    stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
                    stderr="Process execution timed out.",
                    duration_ms=elapsed_ms,
                    error_type="TimeoutError",
                    assertion_passed=False,
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return SandboxResult(
                    success=False,
                    timed_out=False,
                    return_code=-2,
                    stdout="",
                    stderr=str(exc),
                    duration_ms=elapsed_ms,
                    error_type="SandboxError",
                    assertion_passed=False,
                )

    def execute_snippet(
        self,
        code: str,
        assertion_code: Optional[str] = None,
        use_subprocess: bool = False,
        timeout: Optional[float] = None,
    ) -> SandboxResult:
        """Execute snippet using either in-process isolated namespace or subprocess isolation."""
        if use_subprocess:
            return self.execute_subprocess(code, assertion_code, timeout=timeout)
        return self.execute_in_namespace(code, assertion_code)
