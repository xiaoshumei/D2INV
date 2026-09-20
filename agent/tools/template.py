"""
HTML Template Generation Tool for D2INV Agent.

Wraps the InfographicTemplate R3 (Reason→Reflect→Refine) pipeline
into an agent-callable tool. Reads the generated story from session state.
"""

from __future__ import annotations

from agent.tools.base import BaseTool, ToolParam
from agent.session import Session


def _new_llm():
    from tools.llm import LLM

    return LLM()


class EditTemplateTool(BaseTool):

    @property
    def name(self) -> str:
        return "edit_template"

    @property
    def description(self) -> str:
        return (
            "Modify the current HTML infographic template based on a natural-language "
            "user request (layout, colors, styling, section order, etc.). The edited "
            "HTML template is persisted for the next assemble_inv step. Requires "
            "generate_template to have run first."
        )

    @property
    def parameters(self):
        return [
            ToolParam(
                name="dataset_name",
                type="string",
                description="Dataset name (file stem) whose template should be edited, e.g. 'jobs'. If omitted, uses the dataset already loaded in the session.",
                required=False,
            ),
            ToolParam(
                name="prompts",
                type="string",
                description="Natural-language change request for the HTML infographic template.",
                required=True,
            ),
        ]

    def __init__(self, session: Session):
        self._session = session

    @property
    def category(self) -> str:
        return "generation"

    def _execute_impl(self, prompts: str, dataset_name: str = None) -> dict:
        import os

        from agent.tools.cache import read_text
        from api.infographic_template import InfographicTemplate

        dataset_name = dataset_name or self._session.get_state("dataset_name")
        if not dataset_name:
            raise RuntimeError(
                "No dataset specified. Provide dataset_name or load a dataset first."
            )
        self._session.set_state("dataset_name", dataset_name)

        html_template = self._session.get_state("html_template")
        if not html_template:
            html_template = read_text(dataset_name, "infographic_template.html")
        if not html_template:
            raise RuntimeError(
                "No HTML template found on disk or in session. Call generate_template first."
            )
        self._session.set_state("html_template", html_template)

        editor = InfographicTemplate.__new__(InfographicTemplate)
        editor.llm = _new_llm()
        editor.dataset_name = dataset_name
        editor.data_story = self._session.get_state("data_story")
        editor.result = html_template

        edited = editor.edit(prompts)  # returns the updated HTML string

        self._session.set_state("html_template", edited)

        if dataset_name:
            stem = os.path.splitext(os.path.basename(dataset_name))[0]
            dist = os.path.join("results", stem)
            os.makedirs(dist, exist_ok=True)
            with open(
                os.path.join(dist, "infographic_template.html"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(edited)

        return {
            "dataset_name": dataset_name,
            "status": "template_edited",
            "html_length": len(edited),
        }


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