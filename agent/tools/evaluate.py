"""
Self-Evaluation Tool for D2INV Agent.

Wraps the Evaluate module to produce a 5-dimension quality assessment
(engagement, usefulness, legibility, design, aesthetics) of the final INV.
"""

from __future__ import annotations

from agent.tools.base import BaseTool
from agent.session import Session


class EvaluateINVTool(BaseTool):

    @property
    def name(self) -> str:
        return "evaluate_inv"

    @property
    def description(self) -> str:
        return (
            "Self-evaluate the quality of the assembled Interactive Narrative "
            "Visualization across 5 dimensions and generate an HTML radar chart "
            "visualization of the scores. Requires assemble_inv to have run first."
        )

    def __init__(self, session: Session):
        self._session = session

    @property
    def category(self) -> str:
        return "evaluation"

    def _execute_impl(self) -> dict:
        from agent.tools.cache import read_text
        from api.evaluate import Evaluate

        dataset_name = self._session.get_state("dataset_name")
        inv_no_data = self._session.get_state("inv_no_data")

        if dataset_name is None:
            raise RuntimeError(
                "No INV available to evaluate. Run assemble_inv first."
            )

        # Reuse cached evaluation if it already exists on disk.
        cached = read_text(dataset_name, "evaluate.html")
        if cached is not None:
            self._session.set_state("evaluation", cached)
            return {
                "dataset_name": dataset_name,
                "status": "evaluation_complete",
                "html_length": len(cached),
                "cached": True,
            }

        if inv_no_data is None:
            raise RuntimeError(
                "No INV available to evaluate. Run assemble_inv first."
            )

        evaluator = Evaluate(str(dataset_name), inv_no_data)
        evaluation_html = evaluator.run()

        self._session.set_state("evaluation", evaluation_html)

        return {
            "dataset_name": dataset_name,
            "status": "evaluation_complete",
            "html_length": len(evaluation_html),
        }