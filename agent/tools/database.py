"""
SQL Database Tool for D2INV Agent — Phase 2.

Provides read-only SQL query capability against local SQLite databases
(or configurable connections). Uses parameterized queries to prevent
SQL injection where possible. Results are returned as lists of dicts.

Security: Read-only executor by default. Write operations require
explicit opt-in via config.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from agent.tools.base import BaseTool, ToolParam
from agent.session import Session

# Whitelist SQL keywords allowed in read-only mode
_READ_KEYWORDS = frozenset({"SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW", "PRAGMA"})


def _is_read_only(sql: str) -> bool:
    """Naive read-only check based on first keyword."""
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    return first_word in _READ_KEYWORDS


class DatabaseTool(BaseTool):
    """
    Execute SQL queries against a registered database.

    Default engine: SQLite.
    In Phase 2 we keep it simple — one connection string configured
    via session state or constructor argument.
    """

    DEFAULT_TIMEOUT = 10  # seconds

    def __init__(self, session: Session, db_path: Optional[str] = None,
                 read_only: bool = True, timeout: int = DEFAULT_TIMEOUT):
        super().__init__()
        self._session = session
        self._db_path = db_path
        self._read_only = read_only
        self._timeout = timeout

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "execute_sql"

    @property
    def description(self) -> str:
        return (
            "Execute a SQL query against the database. Returns results as a "
            "list of row dictionaries. Supported: SQLite by default. "
            "Only SELECT/WITH/EXPLAIN queries allowed by default for safety. "
            "Use this to explore data, compute aggregations, join tables, etc."
        )

    @property
    def parameters(self):
        return [
            ToolParam(
                name="sql",
                type="string",
                description="SQL query to execute (typically SELECT).",
                required=True,
            ),
            ToolParam(
                name="db_path",
                type="string",
                description="Optional path to SQLite database file. If omitted, uses the session-configured db.",
                required=False,
            ),
        ]

    @property
    def category(self) -> str:
        return "data"

    def _execute_impl(self, sql: str, db_path: Optional[str] = None) -> dict:
        db_path = db_path or self._db_path or self._session.get_state("db_path")

        if not db_path:
            return {
                "success": False,
                "error": "No database configured. Provide db_path or configure via session.",
            }

        if not os.path.exists(db_path):
            return {"success": False, "error": f"Database not found: {db_path}"}

        if self._read_only and not _is_read_only(sql):
            return {
                "success": False,
                "error": "Write/DDL operations are disabled in read-only mode.",
            }

        start_time = time.time()
        try:
            if self._read_only:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=self._timeout)
            else:
                conn = sqlite3.connect(db_path, timeout=self._timeout)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            elapsed = time.time() - start_time

            return {
                "success": True,
                "rows": rows[:200],  # limit to 200 rows for UI
                "total_rows": len(rows),
                "truncated": len(rows) > 200,
                "elapsed_seconds": round(elapsed, 4),
            }
        except Exception as exc:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "error": str(exc),
                "elapsed_seconds": round(elapsed, 4),
            }


class ListTablesTool(BaseTool):
    """List all user tables in the configured database."""

    @property
    def name(self) -> str:
        return "list_db_tables"

    @property
    def description(self) -> str:
        return "List all tables in the configured database. Run this first to understand the schema."

    @property
    def parameters(self):
        return [
            ToolParam(
                name="db_path",
                type="string",
                description="Optional path to SQLite database.",
                required=False,
            ),
        ]

    @property
    def category(self) -> str:
        return "data"

    def __init__(self, session: Session, db_path: Optional[str] = None):
        self._session = session
        self._db_path = db_path

    def _execute_impl(self, db_path: Optional[str] = None) -> dict:
        db_path = db_path or self._db_path or self._session.get_state("db_path")
        if not db_path or not os.path.exists(db_path):
            return {"success": False, "error": f"Database not found: {db_path}"}

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

            # Get column info for each table
            table_details = []
            for table in tables:
                cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
                table_details.append({
                    "name": table,
                    "column_count": len(cols),
                    "columns": [{"name": c[1], "type": c[2]} for c in cols],
                })

            conn.close()
            return {"success": True, "tables": table_details, "count": len(table_details)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}