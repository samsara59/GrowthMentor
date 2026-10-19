
import json
import unittest

from dataclasses import dataclass

from app.assessment.assessor import (
    ExperienceAssessor,
)


@dataclass
class FakeExperience:
    situation: str
    action: str
    outcome: str


class FakeClient:
    """模拟Gemma 3的JSON输出。"""

    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return json.dumps(
            self.response,
            ensure_ascii=False,
        )


class TestExperienceAssessor(unittest.TestCase):

    def setUp(self):

        self.experience = FakeExperience(
            situation="上周模型训练出现错误。",
            action="我检查日志并逐项排查环境依赖。",
            outcome="最终恢复了模型训练。",
        )

    def valid_response(self):

        return {
            "observations": [
                {
                    "description": "逐项检查技术问题",
                    "source_field": "action",
                    "quote": "检查日志并逐项排查",
                }
            ],

            "hypotheses": [
                {
                    "name": "问题分析",
                    "interpretation": (
                        "可能体现有步骤的问题排查行为。"
                    ),
                    "evidence_indices": [0],
                }
            ],

            "follow_up_question": (
                "你能描述另一次独立解决问题的经历吗？"
            ),

            "limitation": (
                "只有一次用户自述，不能确定稳定优势。"
            ),
        }

    def test_valid_assessment(self):

        assessor = ExperienceAssessor(
            FakeClient(self.valid_response())
        )

        result = assessor.assess(
            self.experience
        )

        self.assertEqual(
            len(result.observations),
            1,
        )

        self.assertEqual(
            len(result.hypotheses),
            1,
        )

    def test_reject_fabricated_quote(self):

        response = self.valid_response()

        response["observations"][0]["quote"] = (
            "我领导了十个人的团队"
        )

        assessor = ExperienceAssessor(
            FakeClient(response)
        )

        with self.assertRaises(ValueError):
            assessor.assess(self.experience)

    def test_reject_invalid_evidence_index(self):

        response = self.valid_response()

        response["hypotheses"][0][
            "evidence_indices"
        ] = [9]

        assessor = ExperienceAssessor(
            FakeClient(response)
        )

        with self.assertRaises(ValueError):
            assessor.assess(self.experience)

    def test_accept_insufficient_behavior(self):

        response = {
            "observations": [],
            "hypotheses": [],
            "follow_up_question": (
                "当时你本人具体采取了什么行动？"
            ),
            "limitation": (
                "缺少可观察的具体行动信息。"
            ),
        }

        assessor = ExperienceAssessor(
            FakeClient(response)
        )

        result = assessor.assess(
            self.experience
        )

        self.assertEqual(
            result.hypotheses,
            [],
        )

    def test_reject_hypothesis_without_evidence(self):

        response = self.valid_response()

        response["observations"] = []

        assessor = ExperienceAssessor(
            FakeClient(response)
        )

        with self.assertRaises(ValueError):
            assessor.assess(self.experience)


if __name__ == "__main__":
    unittest.main()
