"""
Clear Results Tool for D2INV Agent.

Deletes all previously generated artifacts (data story, template, INV,
evaluation) for a dataset from the results directory. The agent uses this
tool when a user explicitly asks to regenerate/refresh a dataset's outputs,
so the subsequent generate_* tools recompute everything instead of serving
the cached files.
"""

from __future__ import annotations

from agent.tools.base import BaseTool, ToolParam
from agent.tools.cache import clear_dataset


class ClearResultsTool(BaseTool):

    @property
    def name(self) -> str:
        return "clear_results"

    @property
    def description(self) -> str:
        return (
            "Delete the previously generated results (data story, template, "
            "INV, evaluation) for a dataset from the results directory. Call "
            "this FIRST whenever the user explicitly asks to regenerate, "
            "refresh, or overwrite a dataset's outputs, so the generate_* "
            "tools will recompute them instead of reusing cached files."
        )

    @property
    def parameters(self):
        return [
            ToolParam(
                name="dataset_name",
                type="string",
                description=(
                    "Filename/name of the dataset whose results should be "
                    "deleted, e.g. 'barley' or 'barley.json'."
                ),
                required=True,
            ),
        ]

    def __init__(self, session):
        self._session = session

    @property
    def category(self) -> str:
        return "data"

    def _execute_impl(self, dataset_name: str) -> dict:
        dataset_name = self._session.get_state("dataset_name") or dataset_name
        removed = clear_dataset(dataset_name)
        return {
            "dataset_name": dataset_name,
            "status": "cleared",
            "removed": removed,
            "message": (
                "Existing results deleted." if removed else "No prior results to delete."
            ),
        }