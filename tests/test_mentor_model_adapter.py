import json
import unittest

from app.agent.adaptive_mentor import (
    InvalidModelResponse,
)
from app.core.mentor_model_adapter import (
    MentorModelAdapter,
)


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, system_prompt, user_prompt):
        self.calls.append(
            (system_prompt, user_prompt)
        )
        return self.response


class TestMentorModelAdapter(unittest.TestCase):

    def test_valid_decision(self):
        expected = {
            "action": "respond",
            "message": "Hello",
            "query": "",
            "method_id": None,
            "focus": None,
            "experiment_mode": None,
            "comparison_options": None,
        }

        provider = FakeProvider(
            json.dumps(expected)
        )

        model = MentorModelAdapter(provider)
        result = model.decide("system", "context")

        self.assertEqual(result, expected)
        self.assertEqual(len(provider.calls), 1)

    def test_markdown_json(self):
        provider = FakeProvider(
            '```json\n{"value": 1}\n```'
        )
        model = MentorModelAdapter(provider)

        result = model.generate_json(
            "system",
            "payload",
            {"type": "object"},
        )

        self.assertEqual(result["value"], 1)

    def test_invalid_json(self):
        model = MentorModelAdapter(
            FakeProvider("not json")
        )

        with self.assertRaises(
            InvalidModelResponse
        ):
            model.decide("system", "context")

    def test_non_object(self):
        model = MentorModelAdapter(
            FakeProvider("[]")
        )

        with self.assertRaises(
            InvalidModelResponse
        ):
            model.decide("system", "context")


if __name__ == "__main__":
    unittest.main()
