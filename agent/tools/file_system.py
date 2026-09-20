"""
File System Tool for D2INV Agent — Phase 2.

Secure file operations constrained to the workspace directory
(e.g. project root or a dedicated ``data/`` sandbox). Provides:
  - read text file
  - write text file
  - append text file
  - list directory
  - search file contents (simple grep)

All paths are resolved relative to the configured root directory;
path traversal (``..``) outside the root is rejected.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from agent.tools.base import BaseTool, ToolParam
from agent.session import Session


class FileSystemTool(BaseTool):
    """
    Read/write/list/search files within a restricted workspace root.
    """

    MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB read limit

    def __init__(self, session: Session, root_dir: Optional[str] = None):
        super().__init__()
        self._session = session

        # Root dir: session override > constructor > project dir
        self._root = (root_dir
                      or session.get_state("fs_root")
                      or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._root = os.path.abspath(self._root)

    # ── path safety ───────────────────────────────────────────────────────

    def _resolve(self, path: str) -> str:
        """Resolve a user-supplied path against the root, rejecting traversal."""
        if path is None:
            path = "."
        candidate = os.path.abspath(os.path.join(self._root, path))
        # Ensure candidate is within root
        if not (candidate == self._root or candidate.startswith(self._root + os.sep)):
            raise PermissionError(f"Path escapes workspace root: {path}")
        return candidate

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "file_operation"

    @property
    def description(self) -> str:
        return (
            "Perform a file operation in the workspace. Supports: 'read' text file, "
            "'write' text file (overwrite), 'append' text, 'list' directory, "
            "and 'search' file contents. All paths are relative to the workspace root. "
            "Use this to inspect outputs, save generated HTML/JSON, list result files, etc."
        )

    @property
    def parameters(self):
        return [
            ToolParam(
                name="operation",
                type="string",
                description="One of: read, write, append, list, search.",
                required=True,
            ),
            ToolParam(
                name="path",
                type="string",
                description="File or directory path relative to workspace root.",
                required=True,
            ),
            ToolParam(
                name="content",
                type="string",
                description="Text content for write/append operations.",
                required=False,
            ),
            ToolParam(
                name="pattern",
                type="string",
                description="Regex pattern for search operation.",
                required=False,
            ),
        ]

    @property
    def category(self) -> str:
        return "filesystem"

    def _execute_impl(self, operation: str, path: str, content: str = None,
                      pattern: str = None) -> dict:
        operation = operation.lower().strip()

        if operation == "read":
            full = self._resolve(path)
            if not os.path.isfile(full):
                return {"success": False, "error": f"Not a file: {path}"}
            if os.path.getsize(full) > self.MAX_FILE_SIZE:
                return {"success": False, "error": "File too large to read (>2MB)."}
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            return {"success": True, "path": path, "content": text, "size": len(text)}

        elif operation == "write":
            full = self._resolve(path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content or "")
            return {"success": True, "path": path, "bytes_written": len(content or "")}

        elif operation == "append":
            full = self._resolve(path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "a", encoding="utf-8") as f:
                f.write(content or "")
            return {"success": True, "path": path, "bytes_appended": len(content or "")}

        elif operation == "list":
            full = self._resolve(path)
            if not os.path.isdir(full):
                return {"success": False, "error": f"Not a directory: {path}"}
            entries = []
            for item in sorted(os.listdir(full)):
                item_full = os.path.join(full, item)
                entries.append({
                    "name": item,
                    "type": "dir" if os.path.isdir(item_full) else "file",
                    "size": os.path.getsize(item_full) if os.path.isfile(item_full) else None,
                })
            return {"success": True, "path": path, "entries": entries, "count": len(entries)}

        elif operation == "search":
            full = self._resolve(path)
            if not pattern:
                return {"success": False, "error": "search requires 'pattern'."}
            if not os.path.isdir(full):
                return {"success": False, "error": f"Not a directory: {path}"}

            matches = []
            compiled = re.compile(pattern)
            for root, dirs, files in os.walk(full):
                # skip hidden dirs
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for file in files:
                    fpath = os.path.join(root, file)
                    # skip binary-ish files
                    if file.endswith((".pyc", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip")):
                        continue
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f, start=1):
                                if compiled.search(line):
                                    rel = os.path.relpath(fpath, self._root)
                                    matches.append({"file": rel, "line": i, "text": line.strip()[:200]})
                                    if len(matches) >= 100:
                                        return {"success": True, "pattern": pattern,
                                                "matches": matches, "count": len(matches),
                                                "truncated": True}
                    except Exception:
                        continue
            return {"success": True, "pattern": pattern, "matches": matches,
                    "count": len(matches), "truncated": False}

        else:
            return {"success": False, "error": f"Unknown operation: {operation}. Use read/write/append/list/search."}