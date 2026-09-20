"""
Python REPL Tool for D2INV Agent — Phase 2.

Provides a sandboxed Python execution environment with:
  - stdout/stderr capture
  - execution timeout (wall-clock)
  - restricted builtins (block dangerous calls)
  - persistent variable namespace across invocations
  - configurable max output size

Uses ``subprocess.run`` for cross-platform safety (avoids Windows
multiprocessing bootstrap issues).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import traceback
from string import Template
from typing import Any, Dict, List, Optional

from agent.tools.base import BaseTool, ToolParam
from agent.session import Session


# ---------------------------------------------------------------------------
# Restricted execution script (runs inside child process)
# ---------------------------------------------------------------------------

_RESTRICTED_SCRIPT_TEMPLATE = '''
import json, sys, traceback

# ---- restricted globals ----
_BLOCKED_BUILTINS = {"compile", "eval", "exec", "open", "breakpoint", "input"}

_SAFE_BUILTINS = {
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "classmethod", "complex", "delattr", "dict", "dir",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "globals", "hasattr", "hash", "hex", "id", "int",
    "isinstance", "issubclass", "iter", "len", "list", "locals","map",
    "max", "memoryview", "min", "next", "object", "oct", "ord", "pow",
    "print", "property", "range", "repr", "reversed", "round",
    "set", "setattr", "slice", "sorted", "staticmethod", "str",
    "sum", "super", "tuple", "type", "vars", "zip",
}

def _restricted_import(name, *args, **kwargs):
    blocked_prefixes = ("os", "subprocess", "shutil", "socket", "ctypes", "multiprocessing")
    blocked_exact = {"sys", "signal", "pdb", "code", "codeop", "pty", "fcntl", "posix"}
    if name in blocked_exact:
        raise ImportError(f"Module '{name}' is blocked")
    for prefix in blocked_prefixes:
        if name == prefix or name.startswith(prefix + "."):
            raise ImportError(f"Module '{name}' is blocked")
    return __import__(name, *args, **kwargs)

_restricted_globals = {"__builtins__": {}}
for name in _SAFE_BUILTINS:
    _restricted_globals["__builtins__"][name] = getattr(sys.modules["builtins"], name)
_restricted_globals["__builtins__"]["__import__"] = _restricted_import
_restricted_globals["__name__"] = "__sandbox__"

# ---- load namespace snapshot ----
_namespace = NS_JSON_PLACEHOLDER

# ---- execute user code ----
_restricted_globals.update(_namespace)
try:
    exec(CODE_LITERAL_PLACEHOLDER, _restricted_globals)
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ---- serialize changed variables back ----
_out = {}
for _k, _v in _restricted_globals.items():
    if _k.startswith("__") or _k in _SAFE_BUILTINS:
        continue
    try:
        json.dumps({_k: repr(_v)})
        _out[_k] = repr(_v)
    except Exception:
        _out[_k] = "<unserializable>"

print("__SANDBOX_JSON_OUTPUT__")
print(json.dumps(_out))
print("__END_SANDBOX_JSON_OUTPUT__")
'''

_RESTRICTED_SCRIPT_EXCEPTION_TEMPLATE = '''
import json, sys, traceback

# ---- restricted globals (same as above, minimal reproduction) ----
_BLOCKED_MODULES = {"os", "subprocess", "socket", "ctypes"}
_RSTR_GLOBALS = {
    "__builtins__": {
        k: getattr(sys.modules["builtins"], k)
        for k in {"abs", "bool", "dict", "float", "int", "len", "list",
                  "max", "min", "print", "range", "repr", "round","set",
                  "str", "sum", "tuple", "type", "zip"}
    },
    "__name__": "__sandbox__",
}

try:
    exec({code!r}, _RSTR_GLOBALS)
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''

# ---------------------------------------------------------------------------
# Python REPL Tool
# ---------------------------------------------------------------------------

class PythonREPLTool(BaseTool):
    """
    Sandboxed Python code executor.

    Maintains a persistent namespace across calls so that variables
    defined in one invocation are available in subsequent ones.
    """

    DEFAULT_TIMEOUT = 10
    DEFAULT_MAX_OUTPUT = 5000  # chars

    def __init__(self, session: Session, timeout: int = DEFAULT_TIMEOUT,
                 max_output: int = DEFAULT_MAX_OUTPUT):
        super().__init__()
        self._session = session
        self._timeout = timeout
        self._max_output = max_output

        ns_key = "__python_repl_namespace"
        if not session.has_state(ns_key):
            session.set_state(ns_key, {})
        self._ns_key = ns_key

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "execute_python"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a sandboxed REPL environment. "
            "Variables persist across calls. stdout is captured and returned. "
            "Blocked: os, subprocess, sys, file I/O, socket, eval, exec, compile. "
            f"Timeout: {self._timeout}s. "
            "Use ONLY for data analysis, calculations, pandas/numpy operations, "
            "and data transformation."
        )

    @property
    def parameters(self):
        return [
            ToolParam(
                name="code",
                type="string",
                description="Valid Python code to execute.",
                required=True,
            ),
        ]

    @property
    def category(self) -> str:
        return "computation"

    def _execute_impl(self, code: str) -> dict:
        namespace = self._session.get_state(self._ns_key, {})

        # Serialize namespace to JSON-safe strings for child process injection
        serializable_ns = {}
        for k, v in namespace.items():
            try:
                serializable_ns[k] = repr(v)
            except Exception:
                serializable_ns[k] = f"<{type(v).__name__}>"

        namespace_json = json.dumps(serializable_ns)

        script = _RESTRICTED_SCRIPT_TEMPLATE.replace(
            'NS_JSON_PLACEHOLDER', namespace_json
        ).replace(
            'CODE_LITERAL_PLACEHOLDER', repr(code)
        )

        try:
            proc = subprocess.run(
                [sys.executable, '-c', script],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=os.path.expanduser('~'),  # safe working dir
            )

            stdout = proc.stdout or ''
            stderr = proc.stderr or ''

            # Extract JSON output block
            new_vars = {}
            if '__SANDBOX_JSON_OUTPUT__' in stdout:
                parts = stdout.split('__SANDBOX_JSON_OUTPUT__')
                user_stdout = parts[0].strip()
                json_block = ''
                if '__END_SANDBOX_JSON_OUTPUT__' in stdout:
                    json_block = stdout.split('__END_SANDBOX_JSON_OUTPUT__')[0].split('__SANDBOX_JSON_OUTPUT__')[1].strip()
                try:
                    raw_vars = json.loads(json_block)
                    # parse repr strings back to objects where possible
                    for k, v in raw_vars.items():
                        try:
                            new_vars[k] = eval(v)
                        except Exception:
                            new_vars[k] = v
                except json.JSONDecodeError:
                    stderr += f"\n[JSON parse error on variables: {json_block[:200]}]"
                stdout_text = user_stdout
            else:
                stdout_text = stdout

            # Merge variables into persistent namespace
            namespace.update(new_vars)
            self._session.set_state(self._ns_key, namespace)

            # Truncate
            stdout_text = self._truncate(stdout_text)
            stderr = self._truncate(stderr)

            success = (proc.returncode == 0)

            error_detail = ''
            if not success:
                error_detail = stderr[-500:]

            return {
                "success": success,
                "return_code": proc.returncode,
                "stdout": stdout_text,
                "stderr": stderr,
                "error": error_detail,
                "variable_keys": sorted(new_vars.keys()),
                "namespace_size": len(namespace),
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "return_code": -1,
                "stdout": '',
                "stderr": f'Timeout after {self._timeout}s',
                "error": 'TIMEOUT',
                "variable_keys": [],
                "namespace_size": len(namespace),
            }
        except Exception as exc:
            return {
                "success": False,
                "return_code": -1,
                "stdout": '',
                "stderr": str(exc),
                "error": f'{type(exc).__name__}: {exc}',
                "variable_keys": [],
                "namespace_size": len(namespace),
            }

    def _truncate(self, text: str) -> str:
        if len(text) > self._max_output:
            return text[:self._max_output] + '\n...(truncated)'
        return text