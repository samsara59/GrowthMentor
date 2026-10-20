"""
Synchronous agent message bus.

This module validates and records exchanges.
It does not yet replace the existing LangGraph executor.
"""

from collections import defaultdict
from collections.abc import Callable

from app.brain.messages import AgentMessage


class AgentBus:

    def __init__(self, max_messages_per_request=24):
        if max_messages_per_request < 1:
            raise ValueError("Invalid message limit")

        self.handlers: dict[
            str, Callable[[AgentMessage], AgentMessage | None]
        ] = {}

        self.events: list[AgentMessage] = []
        self.counts = defaultdict(int)
        self.max_messages = max_messages_per_request

    def register(self, agent_id, handler):
        if agent_id in self.handlers:
            raise ValueError("Agent already registered")

        self.handlers[agent_id] = handler

    def dispatch(self, message: AgentMessage):
        key = (message.session_id, message.request_id)

        if self.counts[key] >= self.max_messages:
            raise RuntimeError("Request message limit exceeded")

        if message.recipient not in self.handlers:
            raise KeyError("Unknown recipient")

        self.counts[key] += 1
        self.events.append(message.model_copy(deep=True))

        response = self.handlers[message.recipient](message)

        if response is not None:
            if not isinstance(response, AgentMessage):
                raise TypeError("Handler must return AgentMessage")

            if (
                response.session_id != message.session_id
                or response.request_id != message.request_id
                or response.sender != message.recipient
                or response.parent_id != message.message_id
            ):
                raise ValueError(
                    "Invalid message correlation or sender"
                )

            self.events.append(response.model_copy(deep=True))

        return response

    def trace(self, session_id, request_id):
        return [
            event.model_dump()
            for event in self.events
            if (
                event.session_id == session_id
                and event.request_id == request_id
            )
        ]
