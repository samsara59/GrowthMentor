import json
import unittest

from app.agent.response_validation import (
    GroundedResponseValidator,
)


class FakeProvider:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = 0

    def chat(self, system_prompt, user_prompt):
        self.calls += 1
        return self.responses.pop(0)


class TestResponseValidation(unittest.TestCase):
    def setUp(self):
        self.source = {
            "source": "test_book",
            "location": "section_1",
            "quote": "Practice may improve self-regulation.",
            "original_text": (
                "The chapter states that practice "
                "may improve self-regulation. "
                "Practice may improve self-regulation."
            ),
            "verification": "verified_against_original",
        }

    def test_unverified_source_rejected(self):
        source = dict(self.source)
        source["verification"] = "retrieved_not_checked"

        provider = FakeProvider()
        result = GroundedResponseValidator(provider)(
            "Practice improves skills.", [source]
        )

        self.assertEqual(result["verdict"], "uncertain")
        self.assertEqual(provider.calls, 0)

    def test_fake_quote_rejected(self):
        source = dict(self.source)
        source["quote"] = "FAKE_QUOTE_NOT_IN_SOURCE_987654"

        provider = FakeProvider()
        result = GroundedResponseValidator(provider)(
            "Practice works.", [source]
        )

        self.assertEqual(result["verdict"], "uncertain")
        self.assertEqual(provider.calls, 0)

    def test_supported_claim(self):
        extraction = json.dumps({
            "claims": [{
                "claim": "Practice may improve self-regulation.",
                "source_index": 0,
                "quote": self.source["quote"],
            }]
        })

        nli = json.dumps({
            "verdict": "supported",
            "reason": "Evidence supports claim",
        })

        provider = FakeProvider([extraction, nli])
        result = GroundedResponseValidator(provider)(
            "Practice may improve self-regulation.",
            [self.source],
        )

        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(provider.calls, 2)

    def test_unsupported_claim(self):
        extraction = json.dumps({
            "claims": [{
                "claim": "Practice always works for everyone.",
                "source_index": 0,
                "quote": self.source["quote"],
            }]
        })

        nli = json.dumps({
            "verdict": "uncertain",
            "reason": "Insufficient evidence",
        })

        provider = FakeProvider([extraction, nli])
        result = GroundedResponseValidator(provider)(
            "Practice always works for everyone.",
            [self.source],
        )

        self.assertEqual(result["verdict"], "uncertain")


if __name__ == "__main__":
    unittest.main()
