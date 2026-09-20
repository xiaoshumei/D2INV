"""
INV Assembly Tool for D2INV Agent.

Combines the HTML infographic template with generated chart codes to
produce the final Interactive Narrative Visualization.
"""

from __future__ import annotations

from agent.tools.base import BaseTool
from agent.session import Session


class AssembleINVTool(BaseTool):

    @property
    def name(self) -> str:
        return "assemble_inv"

    @property
    def description(self) -> str:
        return (
            "Combine the HTML template and chart codes into a complete "
            "Interactive Narrative Visualization (INV). Requires "
            "generate_template and generate_charts to have run first."
        )

    def __init__(self, session: Session):
        self._session = session

    @property
    def category(self) -> str:
        return "generation"

    def _execute_impl(self) -> dict:
        from bs4 import BeautifulSoup
        from agent.tools.cache import read_text

        dataset_name = self._session.get_state("dataset_name")
        html_template = self._session.get_state("html_template")
        chart_codes = self._session.get_state("chart_codes")

        # generate_charts may have been run per-piece, leaving only chart_1..chart_N
        # in state instead of a single chart_codes list. Rebuild the full list from
        # the data story so assemble works regardless of how charts were generated.
        if not chart_codes:
            data_story = self._session.get_state("data_story")
            n_pieces = (
                len(data_story.get("story_pieces", []))
                if isinstance(data_story, dict)
                else 0
            )
            chart_codes = [
                self._session.get_state(f"chart_{i + 1}") for i in range(n_pieces)
            ]
            chart_codes = [c for c in chart_codes if c]

        # Reuse the cached assembled INV if it already exists on disk.
        cached = read_text(dataset_name, "inv.html")
        if cached is not None:
            inv_no_data = read_text(dataset_name, "inv_no_data.html") or ""
            self._session.set_state("inv", cached)
            self._session.set_state("inv_no_data", inv_no_data)
            return {
                "dataset_name": dataset_name,
                "status": "inv_assembled",
                "html_length": len(cached),
                "cached": True,
            }

        if not html_template or not chart_codes:
            raise RuntimeError(
                "Missing template or chart codes. Run generate_template and "
                "generate_charts first."
            )

        soup = BeautifulSoup(html_template, "html.parser")

        # Inject ECharts CDN scripts into head
        for src in [
            "https://unpkg.com/echarts@latest/dist/echarts.min.js",
            "https://unpkg.com/echarts@latest/dist/extension/dataTool.min.js",
        ]:
            tag = soup.new_tag("script", src=src)
            soup.head.append(tag)

        # Merge each chart fragment into the template
        for code in chart_codes:
            code_soup = BeautifulSoup(code, "html.parser")
            if code_soup.style:
                soup.head.append(BeautifulSoup(code_soup.style.prettify(), "html.parser"))
            if code_soup.script:
                soup.body.append(BeautifulSoup(code_soup.script.prettify(), "html.parser"))

        inv_no_data = soup.prettify()

        # Inject the actual dataset as a JSON blob for chart consumption
        from tools.utils import filter_dataframe

        df = self._session.get_state("dataframe")
        data_json = filter_dataframe(df)
        tag = soup.new_tag("script")
        tag.string = f"window.data = {data_json}"
        soup.head.append(tag)

        complete_inv = soup.prettify()

        self._session.set_state("inv", complete_inv)
        self._session.set_state("inv_no_data", inv_no_data)

        # Persist to disk via existing write logic
        self._write_result(dataset_name, complete_inv, inv_no_data)

        return {
            "dataset_name": dataset_name,
            "status": "inv_assembled",
            "html_length": len(complete_inv),
        }

    def _write_result(self, dataset_name, complete_inv, inv_no_data):
        import os

        dist = os.path.join("results", os.path.splitext(os.path.basename(dataset_name))[0])
        os.makedirs(dist, exist_ok=True)
        with open(os.path.join(dist, "inv.html"), "w", encoding="utf-8") as f:
            f.write(complete_inv)
        with open(os.path.join(dist, "inv_no_data.html"), "w", encoding="utf-8") as f:
            f.write(inv_no_data)