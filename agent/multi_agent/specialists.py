"""
Specialist Agents — Phase 4: Coder, Designer, Reviewer.

These agents perform concrete work items dispatched by the Orchestrator:
  - CoderAgent:    summarisation, story generation, chart generation, INV assembly
  - DesignerAgent: HTML infographic template design
  - ReviewerAgent: quality review of generated artifacts (story/template/INV)

Each agent maintains no long-lived state except its LLM client; results are
returned as Messages to the Orchestrator.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from tools.utils import postprocess_response
from agent.multi_agent.base import BaseAgent
from agent.multi_agent.protocol import AgentRole, Message, MessageType, result, error


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _llm_json(agent: BaseAgent, system_prompt: str, user_prompt: str,
              temperature: float = 0.35) -> Dict[str, Any]:
    """Call the LLM and parse JSON output.  Raises on failure."""
    completion = agent.llm.client.chat.completions.create(
        model=agent.llm.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = postprocess_response(completion.choices[0].message.content)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Coder Agent
# ---------------------------------------------------------------------------

class CoderAgent(BaseAgent):
    """
    Handles the data/computation pipeline stages:
      - summarise dataset
      - generate data story
      - generate chart code
      - assemble INV
    """

    def __init__(self, llm=None, session=None):
        super().__init__(name="coder", role=AgentRole.CODER, llm=llm)
        self._session = session  # shared Session for stateful tools

    def process(self, msg: Message) -> Optional[Message]:
        if msg.type != MessageType.TASK_REQUEST:
            return None

        action = msg.payload.get("action", "")
        try:
            if action == "summarize":

                data = self._summarize(**msg.payload.get("kwargs", {}))
            elif action == "story":
                data = self._story(**msg.payload.get("kwargs", {}))
            elif action == "charts":
                data = self._charts(**msg.payload.get("kwargs", {}))
            elif action == "assemble":
                data = self._assemble(**msg.payload.get("kwargs", {}))
            else:
                return error(self.name, msg.sender, f"Unknown coder action: {action}",
                             reply_to=msg.msg_id, correlation_id=msg.correlation_id)

            return result(self.name, msg.sender, data,
                          reply_to=msg.msg_id, correlation_id=msg.correlation_id)
        except Exception as exc:
            return error(self.name, msg.sender, f"{type(exc).__name__}: {exc}",
                         reply_to=msg.msg_id, correlation_id=msg.correlation_id)

    # ----- concrete work ------------------------------------------------

    def _summarize(self, dataset_name: str) -> dict:
        from api.summarize import file_summary
        data_summary, df = file_summary(dataset_name)
        # store into session for downstream steps
        if self._session is not None:
            self._session.set_state("dataset_name", dataset_name)
            self._session.set_state("dataframe", df)
            self._session.set_state("data_summary", data_summary)
        return {"status": "summarised", "dataset_name": dataset_name,
                "num_rows": len(df), "num_columns": len(df.columns),
                "columns": list(df.columns)}

    def _story(self) -> dict:
        from api.data_story import DataStory
        if self._session is None:
            raise RuntimeError("CoderAgent needs a Session to generate a story")
        dataset_name = self._session.get_state("dataset_name")
        df = self._session.get_state("dataframe")
        data_summary = self._session.get_state("data_summary")
        story = DataStory(dataset_name, df, data_summary).run_4r()
        self._session.set_state("data_story", story)
        return {"status": "story_generated", "story_title": story.get("story_title"),
                "num_story_pieces": len(story.get("story_pieces", []))}

    def _charts(self) -> dict:
        from api.visualization import visualize_data_story
        from agent.tools.cache import read_text
        if self._session is None:
            raise RuntimeError("CoderAgent needs a Session to generate charts")
        dataset_name = self._session.get_state("dataset_name")
        # If inv.html already exists the charts are already embedded in it —
        # reuse instead of regenerating (regeneration = delete inv.html first).
        if read_text(dataset_name, "inv.html") is not None:
            return {"status": "charts_already_in_inv", "cached": True}
        story = self._session.get_state("data_story")
        summary = self._session.get_state("data_summary")
        codes = visualize_data_story(story, summary)
        self._session.set_state("chart_codes", codes)
        return {"status": "charts_generated", "num_charts": len(codes)}

    def _assemble(self) -> dict:
        from api.inv import INV
        if self._session is None:
            raise RuntimeError("CoderAgent needs a Session to assemble INV")
        dataset_name = self._session.get_state("dataset_name")
        story = self._session.get_state("data_story")
        summary = self._session.get_state("data_summary")
        template = self._session.get_state("html_template")
        df = self._session.get_state("dataframe")
        inv, inv_no_data = INV(dataset_name, story, summary, template, df).run()
        self._session.set_state("inv", inv)
        self._session.set_state("inv_no_data", inv_no_data)
        return {"status": "inv_assembled", "html_length": len(inv)}


# ---------------------------------------------------------------------------
# Designer Agent
# ---------------------------------------------------------------------------

class DesignerAgent(BaseAgent):
    """Designs the infographic HTML template for the data story."""

    def __init__(self, llm=None, session=None):
        super().__init__(name="designer", role=AgentRole.DESIGNER, llm=llm)
        self._session = session

    def process(self, msg: Message) -> Optional[Message]:
        if msg.type != MessageType.TASK_REQUEST:
            return None

        try:
            from api.infographic_template import InfographicTemplate
            dataset_name = self._session.get_state("dataset_name")
            story = self._session.get_state("data_story")
            if dataset_name is None or story is None:
                raise RuntimeError("Designer needs a story first")

            html = InfographicTemplate(str(dataset_name), story).run()
            self._session.set_state("html_template", html)
            return result(self.name, msg.sender,
                          {"status": "template_designed", "html_length": len(html)},
                          reply_to=msg.msg_id, correlation_id=msg.correlation_id)
        except Exception as exc:
            return error(self.name, msg.sender, f"{type(exc).__name__}: {exc}",
                         reply_to=msg.msg_id, correlation_id=msg.correlation_id)


# ---------------------------------------------------------------------------
# Reviewer Agent
# ---------------------------------------------------------------------------

class ReviewerAgent(BaseAgent):
    """
    Reviews generated artifacts for quality issues.

    Two review modes:
      - "story": review the JSON data story for narrative/consistency issues
      - "inv":    self-evaluate the assembled INV (wraps Evaluate)
    """

    def __init__(self, llm=None, session=None):
        super().__init__(name="reviewer", role=AgentRole.REVIEWER, llm=llm)
        self._session = session

    def process(self, msg: Message) -> Optional[Message]:
        if msg.type != MessageType.TASK_REQUEST:
            return None
        mode = msg.payload.get("mode", "story")
        try:
            if mode == "story":
                report = self._review_story()
                return result(self.name, msg.sender, report,
                              reply_to=msg.msg_id, correlation_id=msg.correlation_id)
            elif mode == "inv":
                report = self._evaluate_inv()
                return result(self.name, msg.sender, report,
                              reply_to=msg.msg_id, correlation_id=msg.correlation_id)
            else:
                return error(self.name, msg.sender, f"Unknown review mode: {mode}",
                             reply_to=msg.msg_id, correlation_id=msg.correlation_id)
        except Exception as exc:
            return error(self.name, msg.sender, f"{type(exc).__name__}: {exc}",
                         reply_to=msg.msg_id, correlation_id=msg.correlation_id)

    def _review_story(self) -> dict:
        """Lightweight LLM review of the data story."""
        story = self._session.get_state("data_story")
        if story is None:
            raise RuntimeError("No story to review")

        system_prompt = (
            "You are a careful editor reviewing a data story. Identify issues in "
            "clarity, narrative flow, visualisation appropriateness, and data accuracy "
            "likelihood. Return ONLY JSON: {\"issues\": [...], \"score\": 1-10}."
        )
        user_prompt = f"Review this story:\n{json.dumps(story, ensure_ascii=False)[:4000]}"
        report = _llm_json(self, system_prompt, user_prompt, temperature=0.3)
        report["status"] = "story_reviewed"
        return report

    def _evaluate_inv(self) -> dict:
        """Evaluate INV using 5-dimension scoring (radar chart)."""
        from api.evaluate import Evaluate
        dataset_name = self._session.get_state("dataset_name")
        inv_no_data = self._session.get_state("inv_no_data")
        if dataset_name is None or inv_no_data is None:
            raise RuntimeError("No INV to evaluate")
        evaluation_html = Evaluate(str(dataset_name), inv_no_data).run()
        self._session.set_state("evaluation", evaluation_html)
        return {"status": "inv_evaluated", "html_length": len(evaluation_html)}