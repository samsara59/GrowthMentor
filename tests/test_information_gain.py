
import unittest

from app.assessment.information_gain import (
    Question,
    entropy,
    expected_information_gain,
    select_question,
    update_belief,
)


class TestInformationGain(unittest.TestCase):

    def setUp(self):
        self.prior = [0.5, 0.5]
        self.questions = [
            Question(
                "q1", "问题1",
                [[0.9, 0.1], [0.1, 0.9]],
            ),
            Question(
                "q2", "问题2",
                [[0.6, 0.4], [0.4, 0.6]],
            ),
        ]

    def test_entropy(self):
        self.assertAlmostEqual(
            entropy([0.5, 0.5]), 1.0
        )

    def test_question_selection(self):
        question, gain = select_question(
            self.prior, self.questions
        )
        self.assertEqual(question.question_id, "q1")
        self.assertGreater(gain, 0.5)

    def test_posterior(self):
        result = update_belief(
            self.prior, self.questions[0], 0
        )
        self.assertAlmostEqual(result[0], 0.9)

    def test_no_repeat(self):
        question, _ = select_question(
            self.prior,
            self.questions,
            asked_ids={"q1"},
        )
        self.assertEqual(question.question_id, "q2")


if __name__ == "__main__":
    unittest.main()
