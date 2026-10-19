import unittest

from app.agent.adaptive_mentor import (
    AdaptiveMentor,
    MentorState,
    Decision,
)
from app.domain.contracts import GrowthTask


class FakePlanner:
    model = None

    def plan(self, *args, **kwargs):
        return Decision(
            action="respond",
            message="A factual statement.",
        )


class FakeRegistry:
    def is_bound(self, name):
        return True

    def invoke(self, name, **kwargs):
        return []


class FakeCatalog:
    def all(self):
        return []


class RejectValidator:
    def __init__(self):
        self.calls = 0

    def __call__(self, answer, sources):
        self.calls += 1
        return {
            "verdict": "uncertain",
            "reason": "Not supported",
        }


class TestValidatorWiring(unittest.TestCase):

    def test_validator_parameter_is_stored(self):
        validator = RejectValidator()

        mentor = AdaptiveMentor(
            planner=FakePlanner(),
            registry=FakeRegistry(),
            methods_catalog=FakeCatalog(),
            response_validator=validator,
        )

        self.assertIs(
            mentor.response_validator,
            validator,
        )

    def test_no_evidence_keeps_ordinary_response(self):
        validator = RejectValidator()

        mentor = AdaptiveMentor(
            planner=FakePlanner(),
            registry=FakeRegistry(),
            methods_catalog=FakeCatalog(),
            response_validator=validator,
        )

        state = MentorState(
            task=GrowthTask(
                task_id="test",
                program_id="general",
                goal="Explore a goal",
            )
        )

        new_state, turn = mentor.reply(
            state,
            "I want to reflect on my progress.",
        )

        self.assertEqual(
            turn.message,
            "A factual statement.",
        )

        self.assertEqual(
            validator.calls,
            0,
        )

        self.assertEqual(
            len(new_state.history),
            2,
        )


if __name__ == "__main__":
    unittest.main()
