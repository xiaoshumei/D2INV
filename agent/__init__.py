"""
D2INV Agent Package — Phase 3: Memory-augmented ReAct conversational data visualization agent.

Components:
  - Agent:            Main orchestrator implementing Think→Act→Observe→Decide→Remember loop
  - Planner:          LLM-driven planner that selects the next tool to call
  - Session:          Short-term conversation memory and pipeline state
  - Tools:            Modular tool wrappers for existing pipeline stages
  - Memory:           Long-term semantic memory with vector store (Phase 3)
  - Errors:           Error classification & intelligent recovery (Phase 3)
  - Preferences:      User preference learning & persistence (Phase 3)
"""

from agent.core import Agent, AgentConfig, run_agent
from agent.session import Session, SessionManager
from agent.planner import Planner, PlanStep

# ---- Phase 3 exports ----
from agent.memory import LongTermMemory, SimpleEmbedder, VectorStore
from agent.errors import ErrorClassifier, RecoveryExecutor, RecoveryPlan, ErrorCategory

# ---- Phase 4 exports ----
from agent.multi_agent import Orchestrator

__all__ = [
    "Agent",
    "AgentConfig",
    "run_agent",
    "Session",
    "SessionManager",
    "Planner",
    "PlanStep",
    # Phase 3
    "LongTermMemory",
    "SimpleEmbedder",
    "VectorStore",
    "ErrorClassifier",
    "RecoveryExecutor",
    "RecoveryPlan",
    "ErrorCategory",
    # Phase 4
    "Orchestrator",
]