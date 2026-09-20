"""
Chart Generation Tool for D2INV Agent.

Wraps the Visualization R3 (Reason→Reflect→Refine) pipeline into an
agent-callable tool. Generates ECharts-based code for each story piece
and stores the results in session state.
"""

from __future__ import annotations

from agent.tools.base import BaseTool, ToolParam
from agent.session import Session


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