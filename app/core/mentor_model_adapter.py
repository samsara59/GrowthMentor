"""
OpenAI-compatible model adapter for MentorPlanner.

Supports user-selected hosted or local APIs.
Does not require an Ollama installation.
"""

import json
import re

from app.agent.adaptive_mentor import (
    DECISION_JSON_SCHEMA,
    InvalidModelResponse,
)
from app.core.model_provider import ModelProvider


class MentorModelAdapter:

    def __init__(self, provider: ModelProvider):
        self.provider = provider

    def generate_json(
        self,
        system: str,
        payload: str,
        schema: dict,
    ) -> dict:
        schema_text = json.dumps(
            schema,
            ensure_ascii=False,
        )

        raw = self.provider.chat(
            system
            + "\nReturn only a JSON object matching "
              "this schema:\n"
            + schema_text,
            payload,
        )

        if not isinstance(raw, str):
            raise InvalidModelResponse(
                "Model response must be text"
            )

        cleaned = raw.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise InvalidModelResponse(
                "Model response is not valid JSON"
            ) from exc

        if not isinstance(result, dict):
            raise InvalidModelResponse(
                "Model must return a JSON object"
            )

        return result

    def decide(
        self,
        system: str,
        payload: str,
    ) -> dict:
        return self.generate_json(
            system,
            payload,
            DECISION_JSON_SCHEMA,
        )
