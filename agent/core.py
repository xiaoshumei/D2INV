"""
D2INV Agent Core Module — ReAct Orchestrator (Phase 3).

Implements the main agent loop pattern augmented with:

    THINK → ACT → OBSERVE → REMEMBER → DECIDE → (repeat or finish)

Phase 3 enhancements:
  - LongTermMemory: semantic recall of past interactions before planning
  - Error Recovery: automatic retry for transient errors, fallback strategies
  - Preference Learning: implicit/explicit user preference tracking

Phase 4 enhancements:
  - Multi-Agent Orchestrator: Planner → Coder → Designer → Reviewer → Render Checker pipeline
  - Complex request decomposition into structured task plans
  - Automated quality review and render validation gates

Components:
  - Agent: top-level orchestrator, manages loop and streaming output
  - AgentConfig: tuning parameters for the loop

The Agent uses:
  - planner.Planner       : LLM-driven next-step selection
  - tools.ToolRegistry     : tool lookup & execution
  - session.Session        : short-term memory & pipeline state
  - memory.LongTermMemory  : cross-session semantic memory (Phase 3)
  - errors.RecoveryExecutor: tiered error recovery (Phase 3)
  - preferences.PreferenceLearner: user profiling (Phase 3)
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional

from agent.session import Session, SessionManager
from agent.tools.base import ActionResult, ToolRegistry
from agent.tools import create_tool_registry
from agent.planner import Planner, PlanStep


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Tunable parameters for the Agent loop."""

    max_iterations: int = 30
    """Maximum number of think→act cycles per request."""

    max_consecutive_failures: int = 3
    """Stop early if this many consecutive tool failures occur."""

    yield_step_details: bool = True
    """Whether to stream step-level detail (thought, action, result) to caller."""

    # ---- Phase 3 ----
    enable_long_term_memory: bool = True
    """Inject recalled memories into planner context."""

    memories_to_inject: int = 3
    """Number of top semantic matches to retrieve before planning."""

    enable_error_recovery: bool = True
    """Use RecoveryExecutor for automatic retries."""

    max_auto_retries: int = 3
    """Max retry attempts for transient errors."""

    enable_preference_learning: bool = True
    """Automatically observe user behaviour for preference signals."""

    # ---- Phase 4 ----
    enable_multi_agent: bool = False
    """Use Multi-Agent Orchestrator for complex requests."""


# ---------------------------------------------------------------------------
# Streaming event helpers
# ---------------------------------------------------------------------------

def _event(kind: str, data: Dict[str, Any]) -> str:
    """Wrap a dict into an SSE-style JSON event string."""
    return json.dumps({"event": kind, "data": data}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    D2INV Agent (Phase 3 — memory-augmented ReAct loop).

    Usage:
        agent = Agent(session_mgr, memory_mgr)
        for chunk in agent.stream("Load barley.json and visualize it"):
            send_to_frontend(chunk)
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        config: Optional[AgentConfig] = None,
        # Phase 3 optional dependencies
        long_term_memory: Optional[Any] = None,       # LongTermMemory
        recovery_executor: Optional[Any] = None,      # RecoveryExecutor
        preference_learner: Optional[Any] = None,      # PreferenceLearner
        # Phase 4 optional dependencies
        orchestrator: Optional[Any] = None,             # Orchestrator
    ):
        self.session_mgr = session_manager or SessionManager()
        self.config = config or AgentConfig()
        self._orchestrator_injected = orchestrator

        # Phase 3 modules (lazy-init if not provided)
        self.ltm = long_term_memory
        self.recovery = recovery_executor
        self.prefs = preference_learner

        # Phase 4 orchestrator (lazy-init per session if enabled)
        self._orchestrator_cache = {}

        self._current_session: Optional[Session] = None

    # ── public API ────────────────────────────────────────────────────────

    def stream(
        self,
        user_message: str,
        session_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generator yielding SSE-compatible JSON strings for each step of
        the agent loop. Use this from Flask / FastAPI endpoint.

        Args:
            user_message: the user's natural-language request.
            session_id:   optionally resume an existing session.
        """
        start_time = time.time()

        session = self._ensure_session(session_id)
        session.add_user(user_message)

        # ---- Phase 3: observe user preferences ----
        self._maybe_learn_from_message(user_message)

        # ---- Phase 3: ensure modules are initialised ----
        self._ensure_phase3_modules()

        # ---- Phase 4: multi-agent orchestration path ----
        if self.config.enable_multi_agent:
            orchestrator = self._get_or_create_orchestrator(session)
            outcome = orchestrator.execute(user_message)
            final_answer = self._format_orchestrator_result(outcome)
            session.add_assistant(final_answer)
            yield _event("done", {
                "answer": final_answer,
                "elapsed_seconds": round(time.time() - start_time, 2),
            })
            return

        # Build fresh registry bound to this session
        registry = create_tool_registry(session)
        planner = self._create_planner(registry)

        failures = 0
        last_results: List[Dict] = []

        try:
            for iteration in range(1, self.config.max_iterations + 1):
                # ---- RETRIEVE RELEVANT MEMORIES (Phase 3) ----
                memory_context = self._retrieve_memories(user_message, iteration)

                # ---- THINK ----
                step = planner.plan_next(
                    user_request=user_message,
                    session_state_keys=list(session.state.keys()),
                    session_history=session.get_openai_messages(),
                    last_results=last_results,
                    memory_context=memory_context,
                )

                yield _event("step", {
                    "iteration": iteration,
                    "thought": step.thought,
                    "action": step.tool_name,
                })

                if step.is_done or step.tool_name == "final_answer":
                    final_answer = step.tool_input.get("answer", "")
                    session.add_assistant(final_answer)

                    # ---- Phase 3: remember final answer ----
                    self._remember(
                        f"Q: {user_message[:200]}\nA: {final_answer[:200]}",
                        "conversation",
                        {"session_id": session.session_id},
                    )

                    yield _event("done", {
                        "answer": final_answer,
                        "elapsed_seconds": round(time.time() - start_time, 2),
                    })
                    return

                # ---- ACT (with Phase 3 error recovery wrapping) ----
                try:
                    result = self._execute_with_recovery(
                        registry, step.tool_name, step.tool_input
                    )
                except Exception as exc:
                    yield _event("error", {
                        "message": f"Unrecoverable error executing {step.tool_name}: {exc}",
                    })
                    return

                yield _event("tool_result", {
                    "tool": result.tool_name,
                    "success": result.success,
                    "data": self._safe_data(result.data),
                })

                # Track failure count
                if result.success:
                    failures = 0
                else:
                    failures += 1
                    if failures >= self.config.max_consecutive_failures:
                        yield _event("error", {
                            "message": "Hit consecutive failure limit. Stopping.",
                        })
                        return

                # ---- Phase 3: remember successful tool results ----
                if result.success:
                    self._remember_tool_result(step.tool_name, result.data)

                last_results = [result.to_dict()]

            # exhausted iteration budget
            yield _event("error", {
                "message": (
                    f"Hit max iterations ({self.config.max_iterations}). "
                    "Please simplify your request."
                ),
            })

        except Exception as exc:
            tb = traceback.format_exc()
            yield _event("error", {
                "message": str(exc),
                "traceback": tb,
            })

    # ── Phase 3: internal helpers ─────────────────────────────────────────

    def _ensure_phase3_modules(self):
        """Lazy-initialise optional Phase 3 modules."""
        if self.ltm is None and self.config.enable_long_term_memory:
            from agent.memory.long_term import LongTermMemory
            self.ltm = LongTermMemory()

        if self.recovery is None and self.config.enable_error_recovery:
            from agent.errors.recovery import RecoveryExecutor
            self.recovery = RecoveryExecutor(
                default_max_retries=self.config.max_auto_retries,
            )

        if self.prefs is None and self.config.enable_preference_learning:
            from agent.preferences.learner import PreferenceLearner
            # LTM might still be None if disabled — that's OK, prefs just won't persist
            self.prefs = PreferenceLearner(memory=self.ltm)

    def _get_or_create_orchestrator(self, session: Session):
        """Return cached orchestrator for session, or build a new one."""
        sid = session.session_id
        if sid not in self._orchestrator_cache:
            from agent.multi_agent.orchestrator import Orchestrator
            orch = self._orchestrator_injected or Orchestrator(session=session)
            self._orchestrator_cache[sid] = orch
        return self._orchestrator_cache[sid]

    def _format_orchestrator_result(self, outcome: Dict[str, Any]) -> str:
        """Produce a concise human-readable summary of the orchestrator outcome."""
        if not outcome.get("success", False):
            return f"✗ Task failed: {outcome.get('error', 'unknown error')}"

        artifacts = outcome.get("result", {})
        parts = ["✓ Task completed successfully."]
        if artifacts.get("inv"):
            parts.append("Interactive visualization generated.")
        if artifacts.get("evaluation"):
            parts.append("Quality evaluation available.")
        if artifacts.get("data_story"):
            parts.append("Data story generated.")
        return " ".join(parts)

    def _retrieve_memories(self, query: str, iteration: int) -> str:
        """Return a text block of relevant past memories to inject into planning."""
        if self.ltm is None or not self.config.enable_long_term_memory:
            return ""
        if iteration > 1:
            # Only inject on first iteration to avoid bloat
            return ""

        results = self.ltm.recall(query, top_k=self.config.memories_to_inject)
        if not results:
            return ""

        lines = ["--- Relevant Past Memories ---"]
        for r in results:
            lines.append(
                f"[{r['metadata'].get('type','?')}] "
                f"score={r['score']:.3f} | "
                f"content: {r['metadata'].get('content', r['id'])[:200]}"
            )

        return "\n".join(lines)

    def _remember(self, content: str, memory_type: str, metadata: Optional[Dict] = None):
        if self.ltm is None or not self.config.enable_long_term_memory:
            return
        try:
            self.ltm.remember(content=content, memory_type=memory_type,
                              metadata=metadata)
        except Exception as exc:
            print(f"[LTM] remember failed: {exc}")

    def _remember_tool_result(self, tool_name: str, data: Any):
        """Store successful tool results as memories."""
        if self.ltm is None or not self.config.enable_long_term_memory:
            return

        # Only remember if result data is non-trivial
        if data is None:
            return
        if isinstance(data, dict) and data.get("success") is False:
            return

        summary = self._summarise_for_memory(tool_name, data)
        if summary:
            self._remember(
                content=summary,
                memory_type="tool_result",
                metadata={"tool": tool_name},
            )

    def _execute_with_recovery(self, registry: ToolRegistry,
                               tool_name: str, tool_input: Dict) -> ActionResult:
        """Execute a tool with Phase 3 auto-recovery if enabled."""
        if self.recovery is None or not self.config.enable_error_recovery:
            return registry.execute(tool_name, **tool_input)

        try:
            return self.recovery.execute_with_recovery(
                func=lambda: registry.execute(tool_name, **tool_input),
                context={"tool": tool_name, "input": tool_input},
            )
        except Exception:
            # Recovery failed — return error ActionResult
            return ActionResult(
                tool_name=tool_name,
                success=False,
                error="Recovery executor failed after maximum retries",
            )

    def _maybe_learn_from_message(self, message: str):
        """Implicit preference learning from user message content."""
        if self.prefs is None or not self.config.enable_preference_learning:
            return

        # Language detection via simple heuristic
        if re.search(r'[一-鿿]', message):
            self.prefs.set("language", "zh")
        elif re.search(r'[぀-ヿ]', message):
            self.prefs.set("language", "ja")

        # Chart type mentions
        chart_keywords = {
            "bar": "bar", "柱状图": "bar",
            "line": "line", "折线图": "line",
            "pie": "pie", "饼图": "pie",
            "scatter": "scatter", "散点图": "scatter",
            "map": "map", "地图": "map",
            "heatmap": "heatmap", "热力图": "heatmap",
        }
        lower_msg = message.lower()
        for kw, chart_type in chart_keywords.items():
            if kw in lower_msg:
                self.prefs.observe_chart_choice(chart_type)
                break

    # ── internal (unchanged from Phase 1) ─────────────────────────────────

    def _ensure_session(self, session_id: Optional[str]) -> Session:
        if session_id:
            session = self.session_mgr.get_session(session_id)
            if session is None:
                session = self.session_mgr.create_session(session_id=session_id)
        else:
            session = self.session_mgr.create_session()
        self._current_session = session
        return session

    def _create_planner(self, registry: ToolRegistry):
        from agent.planner import Planner
        return Planner(registry=registry)

    @staticmethod
    def _safe_data(data: Any, max_len: int = 2000) -> Optional[Any]:
        """Truncate serializable data to avoid bloating stream events."""
        if data is None:
            return None
        try:
            s = json.dumps(data, ensure_ascii=False, default=str)
            if len(s) > max_len:
                return s[:max_len] + "...<truncated>"
            return json.loads(s)
        except Exception:
            return str(data)[:max_len]

    @staticmethod
    def _summarise_for_memory(tool_name: str, data: Any) -> str:
        """Create a short text summary of a tool result for memory storage."""
        if not isinstance(data, dict):
            return str(data)[:300]

        parts = [f"[{tool_name}]"]

        # Extract key metadata fields
        for key in ("dataset_name", "status", "count", "num_rows", "num_columns",
                     "story_title", "html_length", "elapsed_seconds"):
            if key in data:
                parts.append(f"{key}={data[key]}")

        # For summarize_dataset, include column info
        if tool_name == "summarize_dataset" and "columns" in data:
            parts.append(f"columns={','.join(data['columns'][:10])}")

        return " ".join(parts)[:500]


# ---------------------------------------------------------------------------
# Helper for direct usage
# ---------------------------------------------------------------------------

def run_agent(user_message: str, session_id: Optional[str] = None) -> str:
    """Synchronous convenience wrapper — collects all stream output into one string."""
    agent = Agent()
    final_answer = ""

    for chunk in agent.stream(user_message, session_id):
        data = json.loads(chunk)
        kind = data.pop("event", "")
        payload = data.get("data", {})

        if kind == "done":
            final_answer = payload.get("answer", "")
        elif kind == "error":
            raise RuntimeError(payload.get("message", "Unknown error"))

    return final_answer