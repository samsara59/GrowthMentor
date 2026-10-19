
import json
import unittest

from app.agent.practice_review import (
    PracticeReviewService,
    descriptive_facts,
)


PLAN = {
    "plan_id": 1,
    "status": "completed",

    "protocol": {
        "option_a": "算法研究",
        "option_b": "产品开发",
        "common_problem": (
            "同一个图像分割应用"
        ),
    },
}


# 全部是虚构的测试数据，
# 不写入真实的个人成长数据库。
LOGS = [
    {
        "log_id": 11,
        "arm": "A",
        "minutes": 45,
        "willingness": 4,
        "engagement": 5,
        "reflection": (
            "主动尝试新的算法改进"
        ),
    },

    {
        "log_id": 12,
        "arm": "B",
        "minutes": 45,
        "willingness": 3,
        "engagement": 4,
        "reflection": (
            "制作可交互界面"
        ),
    },

    {
        "log_id": 13,
        "arm": "B",
        "minutes": 45,
        "willingness": 2,
        "engagement": 3,
        "reflection": (
            "调试交互体验有些疲劳"
        ),
    },

    {
        "log_id": 14,
        "arm": "A",
        "minutes": 45,
        "willingness": 5,
        "engagement": 4,
        "reflection": (
            "继续阅读理论并验证"
        ),
    },
]


REVIEW = {
    "summary": (
        "四条记录初步显示两种活动中"
        "不同的投入体验。"
    ),

    "observed_pattern": (
        "A侧两次主动投入意愿"
        "高于B侧两次自述值。"
    ),

    "observed_log_ids": [
        11, 12, 13, 14,
    ],

    "tentative_interpretation": (
        "目前可能更愿意投入算法研究，"
        "但证据仍然非常有限。"
    ),

    "supporting_log_ids": [
        11, 14,
    ],

    "counter_log_ids": [
        12,
    ],

    "alternative_explanations": [
        "两种任务难度不同",
        "执行先后和疲劳影响",
    ],

    "uncertainties": [
        "长期重复投入是否仍然一致",
    ],

    "next_step": (
        "在任务难度匹配后，"
        "再开展一次小规模实践。"
    ),

    "confirmation_question": (
        "上述解释与你的真实体验一致吗？"
    ),
}


class FakeModel:

    def __init__(self, data):

        self.data = data
        self.calls = []

    def generate_json(
        self,
        system,
        payload,
        schema,
    ):

        self.calls.append(
            json.loads(payload)
        )

        return self.data


class TestPracticeReview(unittest.TestCase):

    def test_valid_review(self):

        fake = FakeModel(REVIEW)

        result = PracticeReviewService(
            fake
        ).analyze(
            PLAN,
            LOGS,
        )

        self.assertEqual(
            result.supporting_log_ids,
            [11, 14],
        )

        self.assertEqual(
            len(
                fake.calls[0][
                    "user_reported_logs"
                ]
            ),
            4,
        )

    def test_reject_nonexistent_evidence(
        self,
    ):

        response = {
            **REVIEW,
            "supporting_log_ids": [999],
        }

        with self.assertRaisesRegex(
            ValueError,
            "不存在",
        ):

            PracticeReviewService(
                FakeModel(response),
                attempts=1,
            ).analyze(
                PLAN,
                LOGS,
            )

    def test_observation_must_include_both_arms(
        self,
    ):

        response = {
            **REVIEW,
            "observed_log_ids": [11, 14],
        }

        with self.assertRaisesRegex(
            ValueError,
            "A和B",
        ):

            PracticeReviewService(
                FakeModel(response),
                attempts=1,
            ).analyze(
                PLAN,
                LOGS,
            )

    def test_need_both_arms(self):

        with self.assertRaisesRegex(
            ValueError,
            "至少",
        ):

            PracticeReviewService(
                FakeModel(REVIEW)
            ).analyze(
                PLAN,
                [LOGS[0]],
            )

    def test_descriptive_statistics(self):

        facts = descriptive_facts(
            LOGS
        )

        self.assertEqual(
            facts["A"]["sessions"],
            2,
        )

        self.assertEqual(
            facts["B"]["mean_willingness"],
            2.5,
        )

    def test_proposed_plan_cannot_be_reviewed(
        self,
    ):

        proposal = {
            **PLAN,
            "status": "proposed",
        }

        with self.assertRaisesRegex(
            ValueError,
            "已确认",
        ):

            PracticeReviewService(
                FakeModel(REVIEW)
            ).analyze(
                proposal,
                LOGS,
            )


if __name__ == "__main__":
    unittest.main()
