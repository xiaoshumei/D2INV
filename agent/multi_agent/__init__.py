"""
D2INV Multi-Agent System — Phase 4.

Exports:
  - Orchestrator: coordinates Planner, Coder, Designer, Reviewer agents
  - Message & AgentRole: inter-agent communication protocol
  - Specialist agents: PlannerAgent, CoderAgent, DesignerAgent, ReviewerAgent
"""

from agent.multi_agent.orchestrator import Orchestrator
from agent.multi_agent.protocol import Message, AgentRole, MessageType
from agent.multi_agent.base import BaseAgent
from agent.multi_agent.planner_agent import PlannerAgent
from agent.multi_agent.specialists import CoderAgent, DesignerAgent, ReviewerAgent

__all__ = [
    "Orchestrator",
    "Message", "AgentRole", "MessageType",
    "BaseAgent",
    "PlannerAgent",
    "CoderAgent", "DesignerAgent", "ReviewerAgent",
]