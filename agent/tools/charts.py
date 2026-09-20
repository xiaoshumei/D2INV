"""
Chart Generation Tool for D2INV Agent.

Wraps the Visualization R3 (Reason→Reflect→Refine) pipeline into an
agent-callable tool. Generates ECharts-based code for each story piece
and stores the results in session state.
"""

from __future__ import annotations

from agent.tools.base import BaseTool, ToolParam
from agent.session import Session


def _new_llm():
    from tools.llm import LLM

    return LLM()


class EditChartTool(BaseTool):

    @property
    def name(self) -> str:
        return "edit_chart"

    @property
    def description(self) -> str:
        return (
            "Modify a single chart's ECharts visualization code for one story piece "
            "based on a natural-language user request (change colors, chart type, "
            "title, axis labels, etc.). Requires generate_charts to have run, or the "
            "chart to already exist (e.g. inside an assembled INV). Provide the "
            "0-based piece_index of the chart to change."
        )

    @property
    def parameters(self):
        return [
            ToolParam(
                name="dataset_name",
                type="string",
                description="Dataset name (file stem), e.g. 'jobs'. Used to locate an existing INV on disk when session state is empty.",
                required=False,
            ),
            ToolParam(
                name="piece_index",
                type="number",
                description="0-based index of the story piece / chart to edit.",
                required=True,
            ),
            ToolParam(
                name="prompts",
                type="string",
                description="Natural-language change request for this chart.",
                required=True,
            ),
        ]

    def __init__(self, session: Session):
        self._session = session

    @property
    def category(self) -> str:
        return "generation"

    def _execute_impl(self, piece_index: int, prompts: str, dataset_name: str = None) -> dict:
        import os
        from api.visualization import Visualization
        from agent.tools.cache import read_text

        dataset_name = dataset_name or self._session.get_state("dataset_name")
        if not dataset_name:
            raise RuntimeError(
                "No dataset specified. Provide dataset_name or load a dataset first."
            )
        self._session.set_state("dataset_name", dataset_name)

        # Pull the existing chart code from session state (either the aggregate
        # chart_codes list or the individual chart_<n> keys).
        chart_codes = self._session.get_state("chart_codes") or []
        if 0 <= piece_index < len(chart_codes) and chart_codes[piece_index]:
            current = chart_codes[piece_index]
        else:
            current = self._session.get_state(f"chart_{piece_index + 1}", "")

        # Fall back: extract the chart fragment from an existing on-disk INV.
        if not current:
            current = _extract_chart_from_html(
                read_text(dataset_name, "inv.html"), piece_index
            )
        if not current:
            raise RuntimeError(
                f"No chart exists for piece index {piece_index}. "
                "Call generate_charts first."
            )

        viz = Visualization.__new__(Visualization)
        viz.llm = _new_llm()
        viz.library = "echarts"
        viz.index = piece_index

        new_code = viz.edit_code(current, prompts)

        # Update both representations so later calls / assembly see the change.
        chart_codes = self._session.get_state("chart_codes") or []
        if piece_index < len(chart_codes):
            chart_codes[piece_index] = new_code
        else:
            chart_codes.append(new_code)
        self._session.set_state("chart_codes", chart_codes)
        self._session.set_state(f"chart_{piece_index + 1}", new_code)

        # Update the assembled INV on disk/session so the edit is reflected in
        # the preview. Fold in any chart still missing from state by re-reading.
        stem = os.path.splitext(os.path.basename(dataset_name))[0]
        inv = self._session.get_state("inv") or read_text(dataset_name, "inv.html")
        if inv:
            inv = _replace_chart_in_html(inv, piece_index, new_code)
            self._session.set_state("inv", inv)
            dist = os.path.join("results", stem)
            os.makedirs(dist, exist_ok=True)
            with open(os.path.join(dist, "inv.html"), "w", encoding="utf-8") as f:
                f.write(inv)

        return {
            "dataset_name": dataset_name,
            "status": "chart_edited",
            "piece_index": piece_index,
            "html_length": len(new_code),
        }


def _extract_chart_from_html(html: str | None, piece_index: int) -> str:
    """
    Reconstruct the inline chart fragment (<style> + <script>) for chart_<index+1>
    from an assembled INV. Returns '' if not found.
    """
    if not html:
        return ""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    style = None
    script = None
    selector = f"#chart_{piece_index + 1}"
    for st in soup.find_all("style"):
        if selector in (st.get_text() or ""):
            style = st
            break
    marker = f"plot_{piece_index + 1}("
    for sc in soup.find_all("script"):
        if marker in (sc.get_text() or ""):
            script = sc
            break
    if style is None and script is None:
        return ""
    parts = []
    for node in (style, script):
        if node is not None:
            parts.append(node.prettify())
    return "\n".join(parts)


def _replace_chart_in_html(html: str, piece_index: int, new_code: str) -> str:
    """
    Within an assembled INV, replace the <style> + <script> pair belonging to
    chart_<index+1> with the edited chart code. Falls back to returning the
    original HTML if the markers cannot be found.
    """
    from bs4 import BeautifulSoup
    import re

    try:
        soup = BeautifulSoup(html, "html.parser")
        new_soup = BeautifulSoup(new_code, "html.parser")

        # Replace the <style> whose selector belongs to this chart.
        for style in soup.find_all("style"):
            stxt = style.get_text() or ""
            if f"#chart_{piece_index + 1}" in stxt:
                replacement = new_soup.style
                if replacement is not None:
                    style.replace_with(replacement)
                else:
                    style.decompose()
                break

        # Replace the <script> containing the chart's plot function.
        marker = f"plot_{piece_index + 1}("
        for script in soup.find_all("script"):
            if marker in (script.get_text() or ""):
                replacement = new_soup.script
                if replacement is not None:
                    script.replace_with(replacement)
                return str(soup)
    except Exception as e:
        print(f"[edit_chart] in-place INV update failed, keeping old INV: {e}")
    return html


class GenerateChartsTool(BaseTool):

    @property
    def name(self) -> str:
        return "generate_charts"

    @property
    def description(self) -> str:
        return (
            "Generate ECharts visualization code for every story piece in the "
            "data story. Each chart is an HTML fragment with a <style> and "
            "<script> block. Requires generate_data_story to have run first."
        )

    @property
    def parameters(self):
        return [
            ToolParam(
                name="piece_index",
                type="number",
                description=(
                    "Optional: 0-based index of a single story piece to render. "
                    "If omitted (or -1), generates charts for all pieces and "
                    "returns a list."
                ),
                required=False,
                default=-1,
            ),
        ]

    def __init__(self, session: Session):
        self._session = session

    @property
    def category(self) -> str:
        return "generation"

    def _execute_impl(self, piece_index: int = -1) -> dict:
        from agent.tools.cache import read_text
        from api.visualization import visualize_data_story

        dataset_name = self._session.get_state("dataset_name")

        # If inv.html already exists, the charts for every story piece are already
        # embedded in it — nothing to regenerate. The planner may call this tool
        # per-piece (piece_index=0,1,...), so the cache check applies regardless
        # of piece_index. To force regeneration the agent calls clear_results
        # first, which deletes inv.html.
        if read_text(dataset_name, "inv.html") is not None:
            return {
                "dataset_name": dataset_name,
                "status": "charts_already_in_inv",
                "message": (
                    "inv.html already exists for this dataset; charts are already "
                    "embedded in it, skipping regeneration."
                ),
                "cached": True,
            }

        data_story = self._session.get_state("data_story")
        data_summary = self._session.get_state("data_summary")

        if data_story is None or data_summary is None:
            raise RuntimeError(
                "No data story or summary found. Call summarize_dataset and "
                "generate_data_story first."
            )

        if piece_index >= 0:
            sub_story = {
                "story_pieces": [data_story["story_pieces"][piece_index]]
            }
            codes = visualize_data_story(sub_story, data_summary)
            key = "chart_" + str(piece_index + 1)
            chart = codes[0] if codes else ""
            self._session.set_state(key, chart)
            # Keep the aggregate chart_codes in sync so assemble_inv can always
            # find the full list even when charts are generated per-piece.
            chart_codes = self._session.get_state("chart_codes") or []
            if piece_index >= len(chart_codes):
                chart_codes.extend([""] * (piece_index + 1 - len(chart_codes)))
            chart_codes[piece_index] = chart
            self._session.set_state("chart_codes", chart_codes)
            return {"piece_index": piece_index, "chart": chart}

        codes = visualize_data_story(data_story, data_summary)
        self._session.set_state("chart_codes", codes)
        return {
            "status": "charts_generated",
            "num_charts": len(codes),
        }