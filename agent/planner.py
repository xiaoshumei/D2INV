"""
Planner module for D2INV Agent — Phase 2 Enhanced.

The planner is responsible for:
  1. Understanding the user's request (goal extraction)
  2. Decomposing the goal into subtasks (plan generation)
  3. Choosing the next action when the agent loops (dynamic routing)

Phase 2 enhancements:
  - Category-grouped tool descriptions for better LLM comprehension
  - Tool usage hints (common patterns)
  - Improved fallback logic
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools.llm import LLM
from tools.utils import postprocess_response
from agent.tools.base import ToolRegistry


# ---------------------------------------------------------------------------
# Planner prompts — Phase 2 with categories
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = """You are the planning module of a powerful data-visualization and analysis agent called D2INV.

Your job is to decide the best next ACTION based on the user's request and the current conversation state.

Available tools are grouped by category:
{tool_descriptions}

Respond ONLY with a valid JSON object in this exact shape:
{{
  "thought": "string, concise reasoning about current state plus why this tool is needed",
  "tool_name": "string, the exact name of ONE tool to call next",
  "tool_input": {{ ... }},
  "is_done": false
}}

RULES:
1. If the request can be fully satisfied with a FINAL response, set "is_done": true, use "tool_name": "final_answer", and place your response in tool_input: {{"answer": "..."}}.
2. If you need information or an intermediate action, set "is_done": false and select exactly ONE tool from the list.
3. ONLY use tools from the provided list.  Never invent tools.
4. Do NOT repeat the same failing tool with unchanged input after it fails once.
5. Always examine the CURRENT STATE KEY VALUES provided below before deciding if a tool has already produced results that can be reused.
6. IMPORTANT — REGENERATION: If the user EXPLICITLY asks to regenerate, refresh, overwrite, or recompute existing results (e.g. "regenerate the story", "refresh the inv", "重新生成"), call "clear_results" FIRST to delete that dataset's existing on-disk results, THEN call the relevant generate_* tools. If no explicit regeneration is requested and a result already exists in state or on disk, reuse it instead of regenerating.
6b. DO NOT use shell/execute_shell or python_repl to list files, walk directories, or verify whether result files exist. The generate_* tools already load existing results from disk themselves and report whether they were cached. If an answer needs file paths, use the ones returned by those tools.
7. For data analysis requests, follow this TYPICAL SEQUENCE:
   - First: list_datasets or summarise_dataset
   - THEN: python_repl (for computations) or execute_sql (for database queries)
   - THEN synthesize findings with final_answer
8. For external knowledge questions, use web_search FIRST, then synthesise.
8. Prefer the most specific tool available. Eg use execute_sql for database queries, python_repl for calculations.
"""


PLAN_CONTEXT_TEMPLATE = """USER REQUEST:
{user_request}

CONVERSATION HISTORY (recent messages):
{history}

CURRENT STATE — these intermediate results are already available (keys):
{state_keys}

LAST TOOL RESULTS (most recent first):
{last_results}

LONG-TERM MEMORY CONTEXT (Phase 3):
{memory_context}

What is the NEXT action to take?"""


# ---------------------------------------------------------------------------
# Planner result
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    thought: str
    tool_name: str
    tool_input: Dict[str, Any] = field(default_factory=dict)
    is_done: bool = False

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PlanStep":
        return PlanStep(
            thought=data.get("thought", ""),
            tool_name=data.get("tool_name", ""),
            tool_input=data.get("tool_input", {}),
            is_done=bool(data.get("is_done", False)),
        )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """
    LLM-driven planner — Phase 2 enhanced.

    Uses category-grouped tool descriptions and richer context to
    drive smarter tool selection decisions across a larger tool set.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        llm: Optional[LLM] = None,
        max_history_messages: int = 24,
    ):
        self.registry = registry
        self.llm = llm or LLM()
        self.max_history_messages = max_history_messages

    # ── helpers ─────────────────────────────────────────────────────────

    def _tool_descriptions(self) -> str:
        """Build grouped human-readable tool listing."""
        return self.registry.get_llm_schemas_grouped()

    def _build_context(
        self,
        user_request: str,
        history: List[Dict[str, str]],
        state_keys: List[str],
        last_results: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> str:
        # truncate history to keep prompt manageable
        recent = history[-self.max_history_messages:]
        history_text = "\n".join(
            f"{m['role']}: {m['content'][:400]}" for m in recent
        )
        if not history_text:
            history_text = "(start of conversation — no prior messages)"

        # show last 5 results with concise info
        results_lines = []
        for r in last_results[-5:]:  # most recent 5
            name = r.get("tool_name", "?")
            ok = r.get("success", False)
            data = r.get("data")
            err = r.get("error", "")
            if isinstance(data, dict):
                info = ", ".join(f"{k}={v}" for k, v in list(data.items())[:5])
            else:
                info = str(data)[:200]
            line = f"  [{name}] success={ok} | {info}"
            if err:
                line += f" | ERROR: {err[:120]}"
            results_lines.append(line)
        results_text = "\n".join(results_lines) if results_lines else "(no tools called yet)"

        return PLAN_CONTEXT_TEMPLATE.format(
            user_request=user_request,
            history=history_text,
            state_keys=", ".join(state_keys) if state_keys else "(none)",
            last_results=results_text,
            memory_context=memory_context if memory_context else "(no long-term memory available)",
        )

    # ── main entry point ─────────────────────────────────────────────────

    def plan_next(
        self,
        user_request: str,
        session_state_keys: List[str],
        session_history: List[Dict[str, str]],
        last_results: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> PlanStep:
        """
        Ask the LLM to produce the next PlanStep from the current context.

        Args:
            user_request: natural language request from user
            session_state_keys: current keys in session state
            session_history: recent conversation messages (OpenAI format)
            last_results: last tool results (most recent first)
            memory_context: Optional Phase 3 long-term memory context string

        Returns:
            PlanStep with is_done=True when a final answer is ready.
        """
        system_prompt = PLAN_SYSTEM_PROMPT.format(
            tool_descriptions=self._tool_descriptions()
        )

        context = self._build_context(
            user_request=user_request,
            history=session_history,
            state_keys=session_state_keys,
            last_results=last_results,
            memory_context=memory_context,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]

        try:
            completion = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                stream=False,
                temperature=0.35,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content
            raw_clean = postprocess_response(raw)
            data = json.loads(raw_clean)
            return PlanStep.from_dict(data)

        except Exception as exc:
            print(f"[Planner] planning failed: {exc}")
            return PlanStep(
                thought="Planner encountered an error.",
                tool_name="final_answer",
                tool_input={
                    "answer": (
                        "I'm sorry, I ran into an internal planning error. "
                        "Please try rephrasing your request."
                    )
                },
                is_done=True,
            )