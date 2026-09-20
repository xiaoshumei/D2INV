"""
Multi-Agent Orchestrator — Phase 4.

The Orchestrator coordinates specialist agents (Planner, Coder, Designer,
Reviewer, RenderChecker) to fulfill complex visualisation requests.

Workflow:
  1. Receive user request → send to PlannerAgent for a task plan
  2. Dispatch planned steps sequentially to the designated specialist
  3. After each step, optionally invoke ReviewerAgent for quality gates
  4. Fan-out to RenderChecker after final assembly for sanity verification
  5. Return the final result back to the main Agent loop
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.llm import LLM
from agent.multi_agent.protocol import (
    AgentRole, Message, MessageType,
    progress, result, error,
)
from agent.multi_agent.base import BaseAgent
from agent.multi_agent.planner_agent import PlannerAgent
from agent.multi_agent.specialists import CoderAgent, DesignerAgent, ReviewerAgent
from agent.session import Session


class Orchestrator(BaseAgent):
    """
    Top-level coordinator that manages the lifecycle of a multi-agent
    task execution.
    """

    def __init__(self, session: Session, llm: Optional[LLM] = None):
        super().__init__(name="orchestrator", role=AgentRole.ORCHESTRATOR, llm=llm)
        self.session = session

        # Create specialist agents
        self.planner = PlannerAgent(llm=llm)
        self.coder = CoderAgent(llm=llm, session=session)
        self.designer = DesignerAgent(llm=llm, session=session)
        self.reviewer = ReviewerAgent(llm=llm, session=session)

        # Wire send callbacks so agents can communicate via orchestrator
        self._agents: List[BaseAgent] = [
            self.planner, self.coder, self.designer, self.reviewer,
        ]
        for a in self._agents:
            a.set_send_callback(self._route_message)

        self._progress_callback = None

    # ------------------------------------------------------------------
    # Message routing — orchestrator acts as the message bus
    # ------------------------------------------------------------------

    def set_progress_callback(self, cb) -> None:
        """Register a callback invoked with (detail: str) on progress events."""
        self._progress_callback = cb

    def _route_message(self, msg: Message) -> None:
        """Receive a message from an agent and deliver to the target."""
        target_name = msg.receiver
        target = self._find_agent(target_name)
        if target is not None:
            target.receive(msg)
            self._notify_progress(f"Routed {msg.type.value} → {target_name}")
        else:
            print(f"[Orchestrator] Unknown recipient: {target_name}")

    def _find_agent(self, name: str) -> Optional[BaseAgent]:
        for a in self._agents:
            if a.name == name:
                return a
        return None

    def _notify_progress(self, detail: str):
        if self._progress_callback is not None:
            self._progress_callback(detail)

    def process(self, msg: Message) -> Optional[Message]:
        """Orchestrator.process handles high-level user messages directly."""
        task = msg.payload.get("task", "")
        if not task:
            return error(self.name, msg.sender, "Orchestrator received empty task",
                         reply_to=msg.msg_id, correlation_id=msg.correlation_id)
        outcome = self.execute(task)
        data = {"success": outcome.get("success", False),
                "result": outcome.get("result", {}),
                "steps_log": outcome.get("steps_log", [])}
        return result(self.name, msg.sender, data,
                      reply_to=msg.msg_id, correlation_id=msg.correlation_id)

    # ------------------------------------------------------------------
    # Main entry point — called from Agent.stream()
    # ------------------------------------------------------------------

    def execute(self, user_message: str) -> Dict[str, Any]:
        """
        Execute a full multi-agent pipeline for the given user request.

        Returns a dict with keys: 'success', 'result', 'steps_log'.
        """
        steps_log: List[Dict] = []
        self._notify_progress("Orchestrator received request; planning phase starting.")

        # ---------- 1. PLAN ----------
        plan_request = Message(
            sender=self.name,
            receiver="planner",
            type=MessageType.TASK_REQUEST,
            payload={"task": user_message, "context": self._build_context()},
            correlation_id="phase4_plan",
        )
        self.planner.receive(plan_request)
        plan_responses = self.planner.process_all()
        if not plan_responses or plan_responses[0].type != MessageType.RESULT:
            return {"success": False, "error": "Planning failed", "steps_log": steps_log}

        plan_data = plan_responses[0].payload.get("result", {})
        steps = plan_data.get("steps", [])
        self._notify_progress(f"Plan ready: {len(steps)} steps")

        # ---------- 2. EXECUTE steps sequentially ----------
        for i, step in enumerate(steps, start=1):
            agent_name = step.get("agent", "coder")
            action = step.get("action", step.get("step", ""))
            self._notify_progress(f"Step {i}/{len(steps)}: {action} → {agent_name}")

            step_msg = Message(
                sender=self.name,
                receiver=agent_name,
                type=MessageType.TASK_REQUEST,
                payload={"action": self._map_step_to_action(step),
                         "kwargs": self._build_kwargs(step),
                         "step_detail": step},
                correlation_id=f"phase4_step_{i}",
            )

            target_agent = self._find_agent(agent_name)
            if target_agent is None:
                steps_log.append({"step": i, "action": action, "status": "unknown_agent"})
                continue

            target_agent.receive(step_msg)
            step_responses = target_agent.process_all()
            ok = any(r.type == MessageType.RESULT for r in step_responses)
            err = next((r for r in step_responses if r.type == MessageType.ERROR), None)
            steps_log.append({
                "step": i, "action": action, "agent": agent_name,
                "status": "success" if ok else "error",
                "error": err.payload.get("payload", {}).get("error", "") if err else "",
            })
            if not ok:
                self._notify_progress(f"⚠ Step {i} failed: {err.payload if err else 'unknown'}")

        # ---------- 3. FAN-OUT: render checker (optional lightweight) ----------
        try:
            from agent.render_checker.validator import quick_html_sanity_check
            final_inv = self.session.get_state("inv", "")
            if final_inv:
                check = quick_html_sanity_check(final_inv)
                steps_log.append({"step": "render_check", "status": "success" if check.get("valid") else "warning",
                                  "check": check})
        except ImportError:
            steps_log.append({"step": "render_check", "status": "skipped",
                              "reason": "render_checker not available"})

        success = all(s.get("status") == "success" for s in steps_log[:-1])
        return {
            "success": success,
            "result": self._collect_final_artifacts(),
            "steps_log": steps_log,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_context(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.session.get_state("dataset_name"),
            "columns": list(getattr(self.session.get_state("dataframe"), "columns", [])),
        }

    def _map_step_to_action(self, step: Dict) -> str:
        """Translate a plan step description to a concrete action token."""
        text = (step.get("step", "") + " " + step.get("expected_output", "")).lower()
        if any(kw in text for kw in ["summar", "summary", "summarize"]):
            return "summarize"
        if any(kw in text for kw in ["story", "data stor"]):
            return "story"
        if any(kw in text for kw in ["template", "infographic", "layout", "design"]):
            return "template"
        if any(kw in text for kw in ["chart", "visual", "visualisation", "visualization", "graph"]):
            return "charts"
        if any(kw in text for kw in ["assembl", "combine", "inv", "build"]):
            return "assemble"
        if any(kw in text for kw in ["review", "evaluat", "check", "quality"]):
            return "review"
        # fallback
        return step.get("agent", "coder") + "_default"

    def _build_kwargs(self, step: Dict) -> Dict[str, Any]:
        kwargs = {"dataset_name": self.session.get_state("dataset_name")}
        if "dataset" in step:
            kwargs["dataset_name"] = step["dataset"]
        return kwargs

    def _collect_final_artifacts(self) -> Dict[str, Any]:
        artifacts = {}
        for key in ("inv", "evaluation", "data_story", "html_template"):
            val = self.session.get_state(key)
            if val is not None:
                artifacts[key] = val[:500] if isinstance(val, str) and len(val) > 500 else val
        return artifacts