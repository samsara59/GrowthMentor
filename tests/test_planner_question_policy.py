
import unittest

from app.agent.adaptive_mentor import (
    Decision,
    MentorPlanner,
)


class FakeModel:

    def decide(self, system, prompt):
        return {
            "action": "ask_user",
            "message": "原始问题是什么？",
            "query": "",
            "method_id": None,
            "focus": "what",
            "experiment_mode": None,
            "comparison_options": None,
        }


class TestQuestionPolicy(unittest.TestCase):

    def test_default_unchanged(self):
        planner = MentorPlanner(FakeModel())

        self.assertIsNone(planner.question_policy)

    def test_optional_policy(self):
        def policy(state, decision):
            return decision.model_copy(
                update={"message": "新的问题是什么？"}
            )

        planner = MentorPlanner(
            FakeModel(),
            question_policy=policy,
        )

        self.assertIs(
            planner.question_policy,
            policy,
        )

    def test_decision_validation(self):
        decision = Decision(
            action="ask_user",
            message="一个问题？",
        )

        changed = decision.model_copy(
            update={"message": "另一个问题？"}
        )

        validated = Decision.model_validate(
            changed
        )

        self.assertEqual(
            validated.message,
            "另一个问题？",
        )


if __name__ == "__main__":
    unittest.main()
