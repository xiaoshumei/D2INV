"""
HTML Template Generation Tool for D2INV Agent.

Wraps the InfographicTemplate R3 (Reason→Reflect→Refine) pipeline
into an agent-callable tool. Reads the generated story from session state.
"""

from __future__ import annotations

from agent.tools.base import BaseTool
from agent.session import Session


class GenerateTemplateTool(BaseTool):

    @property
    def name(self) -> str:
        return "generate_template"

    @property
    def description(self) -> str:
        return (
            "Generate an HTML infographic template based on the data story. "
            "The template includes layout, sections, placeholder chart divs, "
            "and styling. Requires generate_data_story to have run first."
        )

    def __init__(self, session: Session):
        self._session = session

    @property
    def category(self) -> str:
        return "generation"

    def _execute_impl(self) -> dict:
        from agent.tools.cache import read_text
        from api.infographic_template import InfographicTemplate

        dataset_name = self._session.get_state("dataset_name")
        data_story = self._session.get_state("data_story")

        if dataset_name is None:
            raise RuntimeError(
                "No data story found. Call generate_data_story first."
            )

        cached = read_text(dataset_name, "infographic_template.html")
        if cached is not None:
            html_template = cached
        else:
            if data_story is None:
                raise RuntimeError(
                    "No data story found. Call generate_data_story first."
                )
            html_template = InfographicTemplate(str(dataset_name), data_story).run()
        self._session.set_state("html_template", html_template)

        return {
            "dataset_name": dataset_name,
            "status": "template_generated",
            "html_length": len(html_template),
        }