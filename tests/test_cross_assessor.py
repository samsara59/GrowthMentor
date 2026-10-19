
import json
import unittest

from dataclasses import dataclass

from app.assessment.cross_assessor import (
    CrossExperienceAssessor,
)


@dataclass
class FakeExperience:
    situation: str
    action: str
    outcome: str


class FakeClient:

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1

        return json.dumps(
            next(self.responses),
            ensure_ascii=False,
        )


class NoCallClient:

    def generate(self, prompt):
        raise AssertionError(
            "输入不符合条件时不能调用模型"
        )


def sample_experiences():

    return [
        FakeExperience(
            situation="训练模型时出现环境错误。",
            action="我检查日志并逐项验证配置。",
            outcome="最后恢复了模型训练。",
        ),

        FakeExperience(
            situation="项目遇到一项复杂技术问题。",
            action="我梳理问题并逐项测试修复方案。",
            outcome="最终解决了项目中的问题。",
        ),
    ]


def valid_report():

    return {
        "candidates": [
            {
                "name": "有步骤的问题分析",
                "interpretation": (
                    "两条经历都出现逐项排查或验证的行为，"
                    "但仍需进一步核实。"
                ),
                "supports": [
                    {
                        "experience_number": 1,
                        "source_field": "action",
                        "quote": "逐项验证配置",
                    },
                    {
                        "experience_number": 2,
                        "source_field": "action",
                        "quote": "逐项测试修复方案",
                    },
                ],
                "possible_counterevidence": [],
                "uncertainty": (
                    "尚不清楚任务难度和独立完成程度。"
                ),
            }
        ],
        "next_question": (
            "你能描述一次没有顺利解决问题的经历吗？"
        ),
        "limitations": (
            "当前只有两条用户自述，"
            "不能证明稳定优势。"
        ),
    }


class TestCrossAssessor(unittest.TestCase):

    def test_valid_cross_assessment(self):

        client = FakeClient([
            valid_report()
        ])

        assessor = CrossExperienceAssessor(
            client,
            max_attempts=1,
        )

        result = assessor.assess(
            sample_experiences()
        )

        self.assertEqual(
            len(result.candidates),
            1,
        )

        self.assertEqual(
            len(result.candidates[0].supports),
            2,
        )

        self.assertEqual(client.calls, 1)

    def test_requires_two_experiences(self):

        assessor = CrossExperienceAssessor(
            NoCallClient()
        )

        with self.assertRaises(ValueError):
            assessor.assess(
                sample_experiences()[:1]
            )

    def test_same_experience_is_not_repetition(self):

        invalid = valid_report()

        # 两条证据都指向经历1
        invalid["candidates"][0]["supports"][1] = {
            "experience_number": 1,
            "source_field": "action",
            "quote": "检查日志",
        }

        assessor = CrossExperienceAssessor(
            FakeClient([invalid]),
            max_attempts=1,
        )

        with self.assertRaises(ValueError):
            assessor.assess(
                sample_experiences()
            )

    def test_quote_must_match_original(self):

        invalid = valid_report()

        invalid["candidates"][0]["supports"][1][
            "quote"
        ] = "我独立领导了整个研究团队"

        assessor = CrossExperienceAssessor(
            FakeClient([invalid]),
            max_attempts=1,
        )

        with self.assertRaises(ValueError):
            assessor.assess(
                sample_experiences()
            )

    def test_retry_corrects_invalid_reference(self):

        invalid = valid_report()

        invalid["candidates"][0]["supports"][1][
            "experience_number"
        ] = 99

        client = FakeClient([
            invalid,
            valid_report(),
        ])

        assessor = CrossExperienceAssessor(
            client,
            max_attempts=2,
        )

        result = assessor.assess(
            sample_experiences()
        )

        self.assertEqual(client.calls, 2)

        self.assertEqual(
            result.candidates[0]
            .supports[1].experience_number,
            2,
        )

    def test_no_repeat_pattern_is_allowed(self):

        empty_report = {
            "candidates": [],
            "next_question": (
                "可以再描述一个不同情境中的经历吗？"
            ),
            "limitations": (
                "当前没有足够证据支持重复行为模式。"
            ),
        }

        assessor = CrossExperienceAssessor(
            FakeClient([empty_report]),
            max_attempts=1,
        )

        result = assessor.assess(
            sample_experiences()
        )

        self.assertEqual(
            result.candidates,
            [],
        )


if __name__ == "__main__":
    unittest.main()
