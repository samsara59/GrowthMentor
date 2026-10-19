
import json
import unittest

from pydantic import ValidationError

from app.agent.adaptive_mentor import (
    AdaptiveMentor,
    MentorPlanner,
    MentorState,
)

from app.agent.experiment_protocol import (
    ComparisonProtocol,
)

from app.domain.contracts import GrowthTask


def sample_plan():

    return {
        "option_a": "算法研究",
        "option_b": "产品开发",

        "question": (
            "我更愿意长期投入研究还是产品？"
        ),

        "common_problem": (
            "围绕同一冠脉分割项目进行两种工作"
        ),

        "task_a": (
            "研究一个分割误差并尝试改进算法，"
            "记录实验"
        ),

        "output_a": "实验记录与一页结论",

        "task_b": (
            "使用同一算法给用户实现帧选择操作界面"
        ),

        "output_b": "可运行的本地界面",

        "minutes_per_session": 45,

        "constants": [
            "同一数据",
            "同样时间预算",
        ],

        "observations": [
            "主动投入程度",
            "遇阻后继续意愿",
            "过程满意度",
        ],

        "limitations": (
            "难度和疲劳可能影响偏好，"
            "不能据此确定职业。"
        ),

        "confirmation_question": (
            "你愿意尝试这个问题和时长吗？"
        ),
    }


# ============================================
# 1. 模拟Gemma 3两种结构化调用
# ============================================

class FakeModel:

    def __init__(self, decision, plan):

        self.decision = decision
        self.plan = plan
        self.calls = []

    def decide(self, system, payload):

        self.calls.append(
            ("decision", json.loads(payload))
        )

        return self.decision

    def generate_json(
        self,
        system,
        payload,
        schema,
    ):

        self.calls.append(
            ("experiment", json.loads(payload))
        )

        return self.plan


class Registry:

    def is_bound(self, key):
        return False


class Methods:

    def all(self):
        return []


def state():

    return MentorState(
        task=GrowthTask(
            task_id="t",
            program_id="self_discovery",
            goal="找到长期成长方向",
        )
    )


# ============================================
# 2. 实际的Agent逻辑测试
# ============================================

class TestExperimentProtocol(unittest.TestCase):

    def decision(self):

        return {
            "action": "propose_experiment",

            "message": (
                "比较算法研究与产品开发"
            ),

            "query": "",
            "method_id": None,
            "focus": "practice",

            "experiment_mode": "comparison",

            "comparison_options": [
                "算法研究",
                "产品开发",
            ],
        }

    def test_reject_identical_arms(self):

        plan = sample_plan()

        plan["task_b"] = plan["task_a"]

        with self.assertRaises(
            ValidationError
        ):
            ComparisonProtocol.model_validate(
                plan
            )

    def test_two_step_llm_comparison(self):

        model = FakeModel(
            self.decision(),
            sample_plan(),
        )

        agent = AdaptiveMentor(
            MentorPlanner(model),
            Registry(),
            Methods(),
        )

        before = state()

        after, turn = agent.reply(
            before,
            "请设计一周实验比较算法研究和产品开发",
        )

        self.assertEqual(
            turn.action,
            "propose_experiment",
        )

        self.assertEqual(
            len(model.calls),
            2,
        )

        self.assertEqual(
            model.calls[0][0],
            "decision",
        )

        self.assertEqual(
            model.calls[1][0],
            "experiment",
        )

        self.assertEqual(
            turn.experiment_protocol[
                "minutes_per_session"
            ],
            45,
        )

        self.assertIn(
            "A→B→B→A",
            turn.message,
        )

        self.assertEqual(
            len(before.history),
            0,
        )

        self.assertEqual(
            len(after.history),
            2,
        )

    def test_bad_comparison_does_not_claim_success(self):

        plan = sample_plan()

        plan["task_b"] = plan["task_a"]

        model = FakeModel(
            self.decision(),
            plan,
        )

        agent = AdaptiveMentor(
            MentorPlanner(model),
            Registry(),
            Methods(),
        )

        _, turn = agent.reply(
            state(),
            "我想比较两个方向",
        )

        self.assertEqual(
            turn.action,
            "system_error",
        )

        self.assertIsNone(
            turn.experiment_protocol
        )

    def test_legacy_four_field_decision(self):

        # 旧版普通追问仍然能够运行
        model = FakeModel(
            {
                "action": "ask_user",

                "message": (
                    "你最投入的具体问题是什么？"
                ),

                "query": "",
                "method_id": None,
            },
            sample_plan(),
        )

        agent = AdaptiveMentor(
            MentorPlanner(model),
            Registry(),
            Methods(),
        )

        _, turn = agent.reply(
            state(),
            "我不知道喜欢什么",
        )

        self.assertEqual(
            turn.action,
            "ask_user",
        )

        self.assertEqual(
            len(model.calls),
            1,
        )


if __name__ == "__main__":
    unittest.main()
