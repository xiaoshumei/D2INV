"""
Session Management and Short-term Memory for D2INV Agent.

A Session holds:
  - conversation history (messages)
  - intermediate pipeline state (dataset, summary, story, template, inv)
  - user preferences discovered within the session
  - session metadata

This is Phase 1 short-term memory: it persists for the lifetime of a
conversation. Long-term/vector memory arrives in a later phase.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_openai(self) -> Dict[str, str]:
        """Convert to OpenAI chat-message format ({role, content})."""
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    """
    Encapsulates everything the Agent needs to remember about an ongoing
    conversation and the pipeline it is running.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_history: int = 50,
        context_token_budget: int = 4000,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = time.time()
        self.updated_at = time.time()
        self.max_history = max_history
        self.context_token_budget = context_token_budget

        # conversation
        self.messages: List[Message] = []

        # pipeline intermediate state
        self.state: Dict[str, Any] = {}

        # user preferences observed this session
        self.preferences: Dict[str, Any] = {}

    # ── lifecycle helpers ───────────────────────────────────────────────

    def touch(self):
        self.updated_at = time.time()

    # ── messages ────────────────────────────────────────────────────────

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self._trim()
        self.touch()
        return msg

    def add_user(self, content: str) -> Message:
        return self.add_message("user", content)

    def add_assistant(self, content: str) -> Message:
        return self.add_message("assistant", content)

    def add_system(self, content: str) -> Message:
        return self.add_message("system", content)

    def add_tool_result(self, content: str) -> Message:
        return self.add_message("tool", content)

    def get_openai_messages(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """Return messages in OpenAI format, optionally limited to last_n."""
        msgs = self.messages
        if last_n is not None:
            msgs = msgs[-last_n:]
        return [m.to_openai() for m in msgs]

    def _trim(self):
        """Trim history to max_history, keeping the system message if present."""
        if len(self.messages) <= self.max_history:
            return
        # keep any leading system message
        system_msgs = [m for m in self.messages if m.role == "system"]
        other_msgs = [m for m in self.messages if m.role != "system"]
        overflow = len(other_msgs) - self.max_history
        if overflow > 0:
            other_msgs = other_msgs[overflow:]
        self.messages = system_msgs + other_msgs

    # ── state ───────────────────────────────────────────────────────────

    def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value
        self.touch()

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def pop_state(self, key: str, default: Any = None) -> Any:
        return self.state.pop(key, default)

    def has_state(self, key: str) -> bool:
        return key in self.state

    def clear_state(self) -> None:
        self.state.clear()
        self.touch()

    # ── preferences ────────────────────────────────────────────────────

    def set_preference(self, key: str, value: Any) -> None:
        self.preferences[key] = value
        self.touch()

    def get_preference(self, key: str, default: Any = None) -> Any:
        return self.preferences.get(key, default)

    # ── serialization ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "state_keys": list(self.state.keys()),
            "preferences": self.preferences,
        }

    def summary(self) -> str:
        """Human-readable summary for debugging/logging."""
        return (
            f"Session[{self.session_id}] "
            f"messages={len(self.messages)} "
            f"state_keys={list(self.state.keys())} "
            f"prefs={self.preferences}"
        )


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """In-memory session store keyed by session id.

    Phase-1 uses plain dict storage. A persistent store (redis/sqlite/db)
    can replace this in later phases without changing the interface.
    """

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create_session(self, session_id: Optional[str] = None, **kwargs) -> Session:
        session = Session(session_id=session_id, **kwargs)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str, **kwargs) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = self.create_session(session_id=session_id, **kwargs)
        return session

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def __len__(self) -> int:
        return len(self._sessions)