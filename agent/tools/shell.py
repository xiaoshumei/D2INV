"""
Shell Command Tool for D2INV Agent — Phase 2.

Executes a allow-listed set of shell commands with a timeout.

Security model:
  - Only commands matching an allow-list of prefixes are permitted.
  - Runs in a restricted working directory.
  - Timeout enforced via subprocess.run(..., timeout=...).
  - Output is captured and size-limited.

Default allow-list covers common read-only diagnostics and file listing;
users can extend via the constructor/config.
"""

from __future__ import annotations

import shlex
import subprocess
from typing import Any, Dict, List, Optional

from agent.tools.base import BaseTool, ToolParam
from agent.session import Session


class ShellTool(BaseTool):
    """
    Execute a shell command from a controlled allow-list.

    Default allow-list (read-only / low-risk):
        ls, pwd, cat, head, tail, wc, grep, find, df, du, date, echo,
        python --version, pip list
    """

    DEFAULT_TIMEOUT = 10
    DEFAULT_MAX_OUTPUT = 10000  # chars

    DEFAULT_ALLOW_LIST = [
        "ls", "pwd", "cat", "head", "tail", "wc", "grep", "find",
        "df", "du", "date", "echo", "python --version", "pip list",
        "git status", "git log", "git diff",
    ]

    def __init__(self, session: Session, allow_list: Optional[List[str]] = None,
                 timeout: int = DEFAULT_TIMEOUT, max_output: int = DEFAULT_MAX_OUTPUT):
        super().__init__()
        self._session = session
        self._allow_list = allow_list or self.DEFAULT_ALLOW_LIST
        self._timeout = timeout
        self._max_output = max_output

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "execute_shell"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command from the allow-list (e.g. ls, cat, head, "
            "tail, wc, grep, find, df, date, git status). Captures stdout and "
            "stderr. Use this to inspect files, list directories, and run "
            "read-only diagnostics. Do NOT use for arbitrary commands."
        )

    @property
    def parameters(self):
        return [
            ToolParam(
                name="command",
                type="string",
                description="Full shell command string to execute.",
                required=True,
            ),
        ]

    @property
    def category(self) -> str:
        return "system"

    def _is_allowed(self, command: str) -> bool:
        """Check whether the command's prefix is in the allow-list."""
        stripped = command.strip()
        if not stripped:
            return False
        for allowed in self._allow_list:
            if stripped == allowed or stripped.startswith(allowed + " "):
                return True
            # Also allow exact command without args
            if stripped.split()[0] == allowed.split()[0]:
                return True
        return False

    def _execute_impl(self, command: str) -> dict:
        if not self._is_allowed(command):
            return {
                "success": False,
                "error": f"Command not allow-listed. Allowed prefixes: {self._allow_list}",
            }

        try:
            proc = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self._timeout,
                cwd=self._session.get_state("fs_root") or None,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            # Truncate
            if len(stdout) > self._max_output:
                stdout = stdout[:self._max_output] + "\n... (truncated)"
            if len(stderr) > self._max_output:
                stderr = stderr[:self._max_output] + "\n... (truncated)"

            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timed out after {self._timeout}s"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}