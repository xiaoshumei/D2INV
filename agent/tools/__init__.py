"""
D2INV Agent Tools Package — Phase 2.

Exports all agent-callable tools and the tool registry.
Supports Phases 1+2 tools with category-based organisation.
"""

from agent.tools.base import BaseTool, ToolRegistry, Action, ActionResult
from agent.tools.summarize import SummarizeDatasetTool, ListDatasetsTool
from agent.tools.story import GenerateStoryTool, EditStoryTool
from agent.tools.template import GenerateTemplateTool, EditTemplateTool
from agent.tools.charts import GenerateChartsTool, EditChartTool
from agent.tools.assemble import AssembleINVTool
from agent.tools.evaluate import EvaluateINVTool
from agent.tools.clear_results import ClearResultsTool

# ---- Phase 2 tools ----
from agent.tools.python_repl import PythonREPLTool
from agent.tools.web_search import WebSearchTool
from agent.tools.database import DatabaseTool, ListTablesTool
from agent.tools.file_system import FileSystemTool
from agent.tools.shell import ShellTool


def create_tool_registry(session, enable_phase2: bool = True) -> ToolRegistry:
    """
    Create a ToolRegistry pre-loaded with all Phase-1 tools
    and (optionally) Phase-2 tools.

    Args:
        session: The Session instance shared by stateful tools.
        enable_phase2: If False, only Phase-1 tools are loaded.

    Returns:
        Configured ToolRegistry instance ready for agent use.
    """
    registry = ToolRegistry()
    registry.register_many([
        # ---- Phase 1 ----
        ListDatasetsTool(),
        SummarizeDatasetTool(session),
        GenerateStoryTool(session),
        EditStoryTool(session),
        GenerateTemplateTool(session),
        EditTemplateTool(session),
        GenerateChartsTool(session),
        EditChartTool(session),
        AssembleINVTool(session),
        EvaluateINVTool(session),
        ClearResultsTool(session),
    ])

    if enable_phase2:
        registry.register_many([
            PythonREPLTool(session),
            WebSearchTool(session),
            DatabaseTool(session),
            ListTablesTool(session),
            FileSystemTool(session),
            ShellTool(session),
        ])

    return registry


__all__ = [
    "BaseTool", "ToolRegistry", "Action", "ActionResult",
    "create_tool_registry",
]