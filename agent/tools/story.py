"""
Data Story Generation Tool for D2INV Agent.

Wraps the DataStory R4R pipeline (Reason → Review → Reflect → Refine)
into an agent-callable tool. Reads dataset/summary from session state.
"""

from __future__ import annotations

from agent.tools.base import BaseTool
from agent.session import Session


class GenerateStoryTool(BaseTool):

    @property
    def name(self) -> str:
        return "generate_data_story"

    @property
    def description(self) -> str:
        return (
            "Generate a structured data story from the currently loaded dataset. "
            "The story contains a title, subtitle, and 5 story pieces, each with "
            "a narration (data fact), question, and suggested visualization. "
            "Requires summarize_dataset to have run first."
        )

    def __init__(self, session: Session):
        self._session = session

    @property
    def category(self) -> str:
        return "generation"

    def _execute_impl(self) -> dict:
        from agent.tools.cache import read_json
        from api.data_story import DataStory

        dataset_name = self._session.get_state("dataset_name")
        df = self._session.get_state("dataframe")
        data_summary = self._session.get_state("data_summary")

        if dataset_name is None:
            raise RuntimeError("No dataset loaded. Call summarize_dataset first.")

        # If results already exist on disk, serve the cached story instead of
        # calling the LLM again (unless the user explicitly asked to regenerate,
        # in which case ClearResultsTool has already deleted the folder).
        cached = read_json(dataset_name, "data_story.json")
        if cached is not None:
            data_story = cached
        else:
            if df is None or data_summary is None:
                raise RuntimeError(
                    "No dataset loaded. Call summarize_dataset first."
                )
            data_story = DataStory(dataset_name, df, data_summary).run_4r()
        self._session.set_state("data_story", data_story)

        return {
            "dataset_name": dataset_name,
            "story_title": data_story.get("story_title"),
            "story_subtitle": data_story.get("story_subtitle"),
            "num_story_pieces": len(data_story.get("story_pieces", [])),
        }