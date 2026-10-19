
import tempfile
import unittest
from pathlib import Path

from app.agent.adaptive_mentor import (
    AdaptiveMentor,
    MentorState,
    Decision,
)
from app.domain.contracts import GrowthTask
from app.capabilities.source_search import (
    verify_retrieved_sources,
)


QUOTE = (
    "Repeated practice may improve self regulation "
    "under some circumstances and over time."
)


class FakePlanner:
    model = None

    def __init__(self, retrieve=True):
        self.retrieve = retrieve
        self.calls = 0

    def plan(
        self,
        state,
        observations,
        methods,
        retrieval_available,
        trace=None,
    ):
        self.calls += 1

        if self.retrieve and self.calls == 1:
            return Decision(
                action="retrieve_knowledge",
                query="practice",
            )

        return Decision(
            action="respond",
            message="Practice may improve regulation.",
        )


class FakeRegistry:
    def __init__(self, records=None):
        self.records = records or []
        self.calls = 0

    def is_bound(self, name):
        return True

    def invoke(self, name, **kwargs):
        self.calls += 1
        return self.records


class FakeCatalog:
    def all(self):
        return []


class FakeValidator:
    def __init__(self, verdict="supported", fail=False):
        self.verdict = verdict
        self.fail = fail
        self.calls = []

    def __call__(self, answer, sources):
        self.calls.append((answer, sources))

        if self.fail:
            raise RuntimeError("Simulated NLI failure")

        return {
            "verdict": self.verdict,
            "reason": "Mock NLI result",
        }


def make_state():
    return MentorState(
        task=GrowthTask(
            task_id="evidence-test",
            program_id="general",
            goal="Explore learning strategies",
        )
    )


def make_mentor(records, validator, retrieve=True):
    registry = FakeRegistry(records)
    mentor = AdaptiveMentor(
        planner=FakePlanner(retrieve=retrieve),
        registry=registry,
        methods_catalog=FakeCatalog(),
        response_validator=validator,
    )
    return mentor, registry


class TestEvidenceEndToEnd(unittest.TestCase):

    def test_verified_source_reaches_validator(self):
        with tempfile.TemporaryDirectory() as folder:
            books = Path(folder)
            (books / "sample.txt").write_text(
                QUOTE,
                encoding="utf-8",
            )

            records = verify_retrieved_sources(
                [{
                    "source": "sample.txt",
                    "location": "chapter_1",
                    "quote": QUOTE,
                    "verification": "retrieved_not_checked",
                }],
                books,
            )

            self.assertEqual(
                records[0]["verification"],
                "verified_against_original",
            )

            validator = FakeValidator("supported")
            mentor, registry = make_mentor(
                records, validator
            )

            state, turn = mentor.reply(
                make_state(),
                "What does the source say about practice?",
            )

            self.assertEqual(registry.calls, 1)
            self.assertEqual(len(validator.calls), 1)
            self.assertEqual(
                turn.message,
                "Practice may improve regulation.",
            )
            self.assertEqual(
                validator.calls[0][1][0]["verification"],
                "verified_against_original",
            )
            self.assertEqual(len(state.history), 2)

    def test_unverified_source_is_not_approved(self):
        validator = FakeValidator("uncertain")

        records = [{
            "source": "missing.txt",
            "location": "unknown",
            "quote": QUOTE,
            "verification": "retrieved_not_checked",
        }]

        mentor, _ = make_mentor(records, validator)

        _, turn = mentor.reply(
            make_state(),
            "Explain this claim using the textbook.",
        )

        self.assertEqual(len(validator.calls), 1)
        self.assertNotEqual(
            turn.message,
            "Practice may improve regulation.",
        )

    def test_validator_error_is_safe(self):
        validator = FakeValidator(fail=True)

        records = [{
            "source": "sample.txt",
            "location": "chapter_1",
            "quote": QUOTE,
            "original_text": QUOTE,
            "verification": "verified_against_original",
        }]

        mentor, _ = make_mentor(records, validator)

        _, turn = mentor.reply(
            make_state(),
            "Explain the evidence.",
        )

        self.assertEqual(len(validator.calls), 1)
        self.assertNotEqual(
            turn.message,
            "Practice may improve regulation.",
        )

    def test_normal_conversation_without_sources(self):
        validator = FakeValidator()

        mentor, registry = make_mentor(
            [], validator, retrieve=False
        )

        state, turn = mentor.reply(
            make_state(),
            "I want to reflect on my goals.",
        )

        self.assertEqual(
            turn.message,
            "Practice may improve regulation.",
        )
        self.assertEqual(len(validator.calls), 0)
        self.assertEqual(registry.calls, 0)
        self.assertEqual(len(state.history), 2)


    def test_retrieval_empty_blocks_claim(self):
        validator = FakeValidator("uncertain")

        mentor, registry = make_mentor(
            records=[],
            validator=validator,
            retrieve=True,
        )

        _, turn = mentor.reply(
            make_state(),
            "Explain this using source evidence.",
        )

        self.assertEqual(registry.calls, 1)
        self.assertEqual(len(validator.calls), 1)

        self.assertNotEqual(
            turn.message,
            "Practice may improve regulation.",
        )


if __name__ == "__main__":
    unittest.main()
