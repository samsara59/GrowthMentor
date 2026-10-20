"""
Typed inter-agent messages.

Messages are observable events, not hidden reasoning.
"""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field
from uuid import uuid4


MessageType = Literal[
    "task_request",
    "evidence_result",
    "handoff_request",
    "review_result",
    "final_decision",
    "error",
]


class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(
        default_factory=lambda: uuid4().hex
    )
    session_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    sender: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    message_type: MessageType

    parent_id: str | None = None
    payload: dict[str, Any] = Field(
        default_factory=dict
    )
