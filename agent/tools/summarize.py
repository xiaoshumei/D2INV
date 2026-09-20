"""
Data Summarization Tool for D2INV Agent.

Wraps the existing summarize module into an agent-callable tool.
Reads dataset from disk, computes column statistics, stores the
loaded DataFrame and summary in the session for downstream tools.
"""

from __future__ import annotations

import os

import pandas as pd

from agent.tools.base import BaseTool, ToolParam
from agent.session import Session


class SummarizeDatasetTool(BaseTool):

    @property
    def name(self) -> str:
        return "summarize_dataset"

    @property
    def description(self) -> str:
        return (
            "Load and analyze a dataset file from the datasets directory. "
            "Returns column-level statistics: dtype, min, max, mean, std, "
            "unique value counts, and sample values. Use this first whenever "
            "a user asks to explore or visualize data."
        )

    @property
    def parameters(self):
        return [
            ToolParam(
                name="dataset_name",
                type="string",
                description="Filename of the dataset in the datasets/ folder, e.g. 'barley.json'. Must exist on disk.",
                required=True,
            ),
        ]

    def __init__(self, session: Session):
        self._session = session

    @property
    def category(self) -> str:
        return "data"

    def _execute_impl(self, dataset_name: str) -> dict:
        from api.summarize import file_summary

        data_summary, df = file_summary(dataset_name)

        # Normalize the dataset name (strip file extension) so every downstream
        # module builds ./results/<name> paths from the same value, consistent
        # with api/app.py which uses os.path.basename(...).split(".")[0].
        stem = os.path.splitext(dataset_name)[0]

        # Persist into session so later tools can pick them up.
        self._session.set_state("dataset_name", stem)
        self._session.set_state("dataframe", df)
        self._session.set_state("data_summary", data_summary)

        # Persist the summary to disk so the frontend's /api/results/artifacts
        # endpoint can serve it to the Data Summary panel.
        try:
            from api.summarize import write_summary
            write_summary(stem, data_summary)
        except Exception as e:
            print(f"[summarize_dataset] failed to persist summary: {e}")

        # Return a lightweight serializable summary (DataFrame is heavy, and the
        # full column stats live in the persisted data_summary.json on disk).
        return {
            "dataset_name": stem,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
        }


class ListDatasetsTool(BaseTool):

    @property
    def name(self) -> str:
        return "list_datasets"

    @property
    def description(self) -> str:
        return "List all dataset files available in the datasets/ directory."

    def _execute_impl(self) -> list:
        import os
        from tools.config import data_dir

        try:
            files = sorted(os.listdir(data_dir))
            return {"datasets": files, "count": len(files)}
        except Exception as e:
            return {"datasets": [], "count": 0, "error": str(e)}