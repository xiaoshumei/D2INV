"""
Planner Agent — Phase 4.

Responsible for decomposing the user request into a structured, ordered
sequence of subtasks. It reads the conversation context and current state,
then emits a TaskPlan.

Does NOT execute work; it defines WHAT to do (the Coder/Designer do the work).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tools.utils import postprocess_response
from agent.multi_agent.base import BaseAgent
from agent.multi_agent.protocol import AgentRole, Message, MessageType, task_request, result


class PlannerAgent(BaseAgent):
    """Decomposes user requests into subtask plans."""

    def __init__(self, llm=None):
        super().__init__(name="planner", role=AgentRole.PLANNER, llm=llm)

    def process(self, msg: Message) -> Optional[Message]:
        if msg.type != MessageType.TASK_REQUEST:
            return None

        task = msg.payload.get("task", "unknown")
        context = msg.payload.get("context", {})

        plan_json = self._plan(task, context)

        # Send the plan back to the orchestrator.
        return result(
            sender=self.name,
            receiver=msg.sender,
            data=plan_json,
            reply_to=msg.msg_id,
            correlation_id=msg.correlation_id,
        )

    # ------------------------------------------------------------------
    def _plan(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a plan via LLM. Returns a dict with steps."""
        system_prompt = """You are a planning specialist for data visualization tasks.

Given a high-level request, break it down into 3-8 ordered steps.
Each step must specify:
  - "step": a short imperative phrase
  - "agent": which agent should handle it ("coder"|"designer"|"render_checker")
  - "expected_output": what artifact or result this step should produce

Return ONLY valid JSON in this shape:
{"steps": [{"step": "...", "agent": "...", "expected_output": "..."}]}"""

        user_prompt = f"REQUEST: {task}\n\nCONTEXT: {json.dumps(context, ensure_ascii=False)[:2000]}"

        try:
            completion = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                temperature=0.35,
                response_format={"type": "json_object"},
            )
            raw = postprocess_response(completion.choices[0].message.content)
            plan = json.loads(raw)
            plan.setdefault("task", task)
            plan.setdefault("steps", [])
            return plan
        except Exception as exc:
            # Fallback: a reasonable default plan.
            print(f"[PlannerAgent] LLM plan failed: {exc}")
            return {
                "task": task,
                "steps": [
                    {"step": "Summarize dataset", "agent": "coder", "expected_output": "data summary"},
                    {"step": "Generate data story", "agent": "coder", "expected_output": "JSON story"},
                    {"step": "Generate infographic template", "agent": "designer", "expected_output": "HTML"},
                    {"step": "Generate charts", "agent": "coder", "expected_output": "chart HTML fragments"},
                    {"step": "Assemble INV", "agent": "coder", "expected_output": "final HTML"},
                    {"step": "Render check", "agent": "render_checker", "expected_output": "validation report"},
                ],
            }