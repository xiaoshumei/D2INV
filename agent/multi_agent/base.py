"""
Base Agent class for D2INV Multi-Agent System — Phase 4.

Provides:
  - message queue (inbox)
  - message sending to orchestrator/router
  - lifecycle hooks (setup / teardown)
  - LLM access shared across agents
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable

from tools.llm import LLM
from agent.multi_agent.protocol import Message, MessageType, AgentRole


class BaseAgent(ABC):
    """Abstract base for all specialist agents."""

    def __init__(self, name: str, role: AgentRole, llm: Optional[LLM] = None):
        self.name = name
        self.role = role
        self.llm = llm or LLM()
        self.inbox: List[Message] = []
        self._send_callback: Optional[Callable[[Message], None]] = None

    # ------------------------------------------------------------------
    # communication hooks
    # ------------------------------------------------------------------

    def set_send_callback(self, cb: Callable[[Message], None]) -> None:
        """Register a callback used to dispatch messages to other agents."""
        self._send_callback = cb

    def send(self, msg: Message) -> None:
        """Send a message via the registered callback."""
        if self._send_callback is None:
            raise RuntimeError(f"Agent '{self.name}' has no send callback set")
        self._send_callback(msg)

    def receive(self, msg: Message) -> None:
        """Receive a message into this agent's inbox."""
        self.inbox.append(msg)

    def pending_messages(self) -> List[Message]:
        return self.inbox

    def clear_inbox(self) -> None:
        self.inbox.clear()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Called once before the agent starts processing."""
        pass

    def teardown(self) -> None:
        """Called when the agent is shutting down."""
        pass

    # ------------------------------------------------------------------
    # processing
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, msg: Message) -> Optional[Message]:
        """Handle an incoming message. Return a response Message or None."""
        ...

    def process_all(self) -> List[Message]:
        """Process all pending inbox messages. Returns response messages."""
        responses = []
        for msg in self.inbox[:]:
            resp = self.process(msg)
            if resp:
                responses.append(resp)
        self.inbox.clear()
        return responses