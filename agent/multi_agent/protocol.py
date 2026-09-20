"""
Message Protocol for D2INV Multi-Agent System — Phase 4.

Defines a unified, JSON-serialisable message envelope for inter-agent
communication. The protocol is transport-agnostic and supports:
  - task assignment / delegation
  - status updates / progress reports
  - data payload transfer (with optional type hints)
  - final results and error reporting
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class AgentRole(Enum):
    """Roles in the multi-agent system."""

    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    CODER = "coder"
    DESIGNER = "designer"
    REVIEWER = "reviewer"
    RENDER_CHECKER = "render_checker"


class MessageType(Enum):
    """Semantic message types exchanged between agents."""

    TASK_REQUEST = "task_request"      # assign work to an agent
    TASK_ACCEPT = "task_accept"        # agent acknowledges task
    PROGRESS = "progress"              # intermediate update
    RESULT = "result"                  # completed work product
    ERROR = "error"                    # failure with reason
    CANCEL = "cancel"                  # terminate an in-flight task
    PING = "ping"                      # liveness check
    PONG = "pong"                      # liveness reply


@dataclass
class Message:
    """A single inter-agent message envelope."""

    sender: str                     # agent name/role string, e.g. "planner"
    receiver: str                   # intended recipient, e.g. "coder" or "*"
    type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None  # msg_id this is in response to
    correlation_id: Optional[str] = None  # groups messages of a single task
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dict (MessageType → string)."""
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Message":
        """Deserialise from a dict."""
        return Message(
            sender=data["sender"],
            receiver=data["receiver"],
            type=MessageType(data["type"]),
            payload=data.get("payload", {}),
            msg_id=data.get("msg_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            reply_to=data.get("reply_to"),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def task_request(sender: str, receiver: str, task: str,
                 payload: Optional[Dict] = None,
                 correlation_id: Optional[str] = None) -> Message:
    """Create a TASK_REQUEST message."""
    d = {"task": task}
    if payload:
        d.update(payload)
    return Message(sender=sender, receiver=receiver,
                   type=MessageType.TASK_REQUEST, payload=d,
                   correlation_id=correlation_id)


def result(sender: str, receiver: str, data: Any,
           reply_to: Optional[str] = None,
           correlation_id: Optional[str] = None) -> Message:
    """Create a RESULT message carrying output data."""
    return Message(sender=sender, receiver=receiver,
                   type=MessageType.RESULT,
                   payload={"result": data},
                   reply_to=reply_to,
                   correlation_id=correlation_id)


def error(sender: str, receiver: str, reason: str,
          reply_to: Optional[str] = None,
          correlation_id: Optional[str] = None) -> Message:
    """Create an ERROR message."""
    return Message(sender=sender, receiver=receiver,
                   type=MessageType.ERROR,
                   payload={"error": reason},
                   reply_to=reply_to,
                   correlation_id=correlation_id)


def progress(sender: str, receiver: str, detail: str, percent: float = 0.0,
             correlation_id: Optional[str] = None) -> Message:
    """Create a PROGRESS message."""
    return Message(sender=sender, receiver=receiver,
                   type=MessageType.PROGRESS,
                   payload={"detail": detail, "percent": percent},
                   correlation_id=correlation_id)