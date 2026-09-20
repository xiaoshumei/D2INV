"""
Data Story Generation Tool for D2INV Agent.

Wraps the DataStory R4R pipeline (Reason → Review → Reflect → Refine)
into an agent-callable tool. Reads dataset/summary from session state.
"""

from __future__ import annotations

from agent.tools.base import BaseTool, ToolParam
from agent.session import Session


def _new_llm():
    from tools.llm import LLM

    return LLM()


class EditStoryTool(BaseTool):

    @property
    def name(self) -> str:
        return "edit_data_story"

    @property
    def description(self) -> str:
        return (
            "Modify the current data story (title, subtitle, or any story piece's "
            "narration/question/visualization) based on a natural-language user request. "
            "Requires generate_data_story to have run first. The edited JSON is "
            "persisted and used by subsequent template/chart generation."
        )

    @property
    def parameters(self):
        return [
            ToolParam(
                name="dataset_name",
                type="string",
                description="Dataset name (file stem) whose data story should be edited, e.g. 'jobs'. If omitted, uses the dataset already loaded in the session.",
                required=False,
            ),
            ToolParam(
                name="prompts",
                type="string",
                description="Natural-language change request for the data story.",
                required=True,
            ),
        ]

    def __init__(self, session: Session):
        self._session = session

    @property
    def category(self) -> str:
        return "generation"

    def _execute_impl(self, prompts: str, dataset_name: str = None) -> dict:
        import json
        import os

        from api.data_story import DataStory
        from agent.tools.cache import read_json

        # Resolve the dataset: explicit param > session state.
        dataset_name = dataset_name or self._session.get_state("dataset_name")
        if not dataset_name:
            raise RuntimeError(
                "No dataset specified. Provide dataset_name or load a dataset first."
            )
        self._session.set_state("dataset_name", dataset_name)

        data_story = self._session.get_state("data_story")
        if data_story is None or not data_story.get("story_pieces"):
            data_story = read_json(dataset_name, "data_story.json")
        if data_story is None or not data_story.get("story_pieces"):
            raise RuntimeError(
                "No data story found on disk or in session. Call generate_data_story first."
            )
        self._session.set_state("data_story", data_story)

        # Feed the *current* story back through DataStory.edit(), which asks the
        # LLM to return an updated JSON story reflecting the user's request.
        editor = DataStory.__new__(DataStory)
        editor.llm = _new_llm()
        editor.dataset_name = dataset_name
        editor.result = json.dumps(data_story, ensure_ascii=False)

        # edit() returns the raw JSON string (already post-processed); parse it.
        edited = editor.edit(prompts)
        try:
            parsed = json.loads(edited)
        except Exception:
            raise RuntimeError("Edited data story was not valid JSON.") from None

        if not parsed.get("story_pieces"):
            raise RuntimeError("The edited data story is invalid (missing story_pieces).")

        self._session.set_state("data_story", parsed)

        if dataset_name:
            stem = os.path.splitext(os.path.basename(dataset_name))[0]
            dist = os.path.join("results", stem)
            os.makedirs(dist, exist_ok=True)
            with open(os.path.join(dist, "data_story.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(parsed, ensure_ascii=False, indent=2))

        return {
            "dataset_name": dataset_name,
            "status": "story_edited",
            "story_title": parsed.get("story_title"),
            "num_story_pieces": len(parsed.get("story_pieces", [])),
        }


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