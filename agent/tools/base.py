"""
Tool Interface and Registry for D2INV Agent.

Provides:
- Abstract BaseTool with standardized schema
- ToolRegistry for dynamic tool registration and discovery
- Action/Result data classes for structured tool I/O
"""

from __future__ import annotations

import json
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Structured I/O
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """A planned action the agent intends to execute."""

    tool_name: str
    tool_input: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # why this action was chosen (chain-of-thought)


@dataclass
class ActionResult:
    """Structured result returned by a tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Parameter Specification (for LLM tool-use awareness)
# ---------------------------------------------------------------------------

@dataclass
class ToolParam:
    name: str
    type: str  # "string" | "number" | "boolean" | "object" | "array"
    description: str
    required: bool = True
    default: Any = None


# ---------------------------------------------------------------------------
# Base Tool
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """
    Abstract base for every agent tool.

    Subclasses must implement:
      - name          : unique tool identifier
      - description   : what the tool does (shown to the LLM)
      - parameters    : input specification
      - _execute_impl : actual logic

    The run() wrapper adds structured error handling and logging.
    """

    # ── metadata (override in subclass) ──────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name, e.g. 'summarize_data'."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the LLM."""

    @property
    def parameters(self) -> List[ToolParam]:
        """Input parameter specification."""
        return []

    @property
    def category(self) -> str:
        """Tool category for grouping and LLM context.  Default: 'general'."""
        return "general"

    # ── execution ─────────────────────────────────────────────────────────

    def run(self, **kwargs) -> ActionResult:
        """Execute the tool with structured error handling."""
        try:
            result_data = self._execute_impl(**kwargs)
            return ActionResult(
                tool_name=self.name,
                success=True,
                data=result_data,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            return ActionResult(
                tool_name=self.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"traceback": tb},
            )

    @abstractmethod
    def _execute_impl(self, **kwargs) -> Any:
        """Implement the core logic.  Return any JSON-serializable data."""

    # ── LLM-awareness helpers ────────────────────────────────────────────

    def to_llm_schema(self) -> Dict[str, Any]:
        """Generate an OpenAI-compatible function/tool schema."""
        props = {}
        required = []
        for p in self.parameters:
            props[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Central registry for tools the Agent can use.

    Supports:
      - registering tools by name
      - retrieving tool schema list for LLM function-calling
      - executing a tool by name with kwargs
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    # ── registration ─────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.  Overwrites if name already exists."""
        name = tool.name
        if name in self._tools:
            print(f"[ToolRegistry] ⚠ Overwriting tool '{name}'")
        self._tools[name] = tool
        print(f"[ToolRegistry] Registered tool: {name}")

    def register_many(self, tools: List[BaseTool]) -> None:
        for t in tools:
            self.register(t)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ── lookup ───────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_llm_schemas(self) -> List[Dict[str, Any]]:
        """Return an OpenAI function-calling schema list for registered tools."""
        return [t.to_llm_schema() for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> ActionResult:
        """Run a registered tool by name; returns ActionResult even on errors."""
        tool = self._tools.get(name)
        if tool is None:
            return ActionResult(
                tool_name=name,
                success=False,
                error=f"Unknown tool: '{name}'. Available: {self.list_tools()}",
            )
        return tool.run(**kwargs)

    def execute_action(self, action: Action) -> ActionResult:
        """Execute from an Action dataclass."""
        return self.execute(action.tool_name, **action.tool_input)

    def get_categories(self) -> Dict[str, List[str]]:
        """Return a mapping of category -> list of tool names."""
        cats: Dict[str, List[str]] = {}
        for name, tool in self._tools.items():
            cat = tool.category
            cats.setdefault(cat, []).append(name)
        return cats

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        return [t for t in self._tools.values() if t.category == category]

    def get_llm_schemas_grouped(self) -> str:
        """Return a human-readable grouped description of tools for the planner."""
        lines = []
        for cat, tools in sorted(self.get_categories().items()):
            lines.append(f"\n[{cat.upper()} TOOLS]")
            for name in sorted(tools):
                t = self._tools[name]
                params = ", ".join(p.name for p in t.parameters)
                lines.append(f"  - {name}({params}): {t.description}")
        return "\n".join(lines)