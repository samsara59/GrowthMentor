
"""第66课：测试任务切换、探索焦点和教材证据。"""

import json
import unittest

from app.agent.adaptive_mentor import (
    AdaptiveMentor,
    MentorPlanner,
    MentorState,
)

from app.agent.dialogue_strategy import (
    build_growth_view,
    guide_for,
)

from app.domain.contracts import GrowthTask


# ============================================
# 1. 模拟大语言模型
# ============================================

class FakeModel:

    def __init__(self, outputs):

        self.outputs = list(outputs)
        self.contexts = []

    def decide(self, system, payload):

        self.contexts.append(
            json.loads(payload)
        )

        return self.outputs.pop(0)


# ============================================
# 2. 模拟能力注册表
# ============================================

class FakeRegistry:

    def __init__(self, handler=None):

        self.handler = handler
        self.calls = []

    def is_bound(self, key):

        return (
            key == "knowledge.retrieve"
            and self.handler is not None
        )

    def invoke(self, key, **kwargs):

        self.calls.append(
            (key, kwargs)
        )

        return self.handler(**kwargs)


class FakeCatalog:

    def all(self):
        return []


def state(
    program="self_discovery",
    goal="探索自己",
):

    return MentorState(
        task=GrowthTask(
            task_id="t1",
            program_id=program,
            goal=goal,
        )
    )


# ============================================
# 3. 七项核心测试
# ============================================

class DialogueStrategyTests(unittest.TestCase):

    def agent(
        self,
        outputs,
        registry=None,
    ):

        model = FakeModel(outputs)

        return (
            AdaptiveMentor(
                MentorPlanner(model),
                registry or FakeRegistry(),
                FakeCatalog(),
            ),
            model,
        )

    # ========================================
    # 测试1：不同Program使用不同目标
    # ========================================

    def test_program_guide_is_not_fixed_questionnaire(
        self,
    ):

        self.assertIn(
            "WHAT",
            guide_for("self_discovery"),
        )

        self.assertIn(
            "备选方案",
            guide_for("reflective_decision"),
        )

        self.assertNotIn(
            "WHAT",
            guide_for("historical_learning"),
        )

        self.assertNotIn(
            "WHAT",
            guide_for("unknown_program"),
        )

    # ========================================
    # 测试2：焦点由LLM输出
    # ========================================

    def test_llm_decides_focus_not_code(self):

        agent, model = self.agent([
            {
                "action": "ask_user",

                "message": (
                    "哪一类事情你愿意主动探索？"
                ),

                "focus": "what",
            }
        ])

        s = state()

        s.confirmed_notes = [
            {
                "category": "strength",
                "text": "我擅长排查训练报错",
            }
        ]

        result, turn = agent.reply(
            s,
            "我擅长写代码，但不知道喜欢什么",
        )

        self.assertEqual(
            turn.focus,
            "what",
        )

        self.assertEqual(
            turn.action,
            "ask_user",
        )

        self.assertIn(
            "我擅长排查训练报错",
            str(
                model.contexts[0][
                    "confirmed_long_term_notes"
                ]
            ),
        )

        self.assertEqual(
            len(s.history),
            0,
        )

        self.assertEqual(
            len(result.history),
            2,
        )

    # ========================================
    # 测试3：允许直接总结而非一直追问
    # ========================================

    def test_user_asks_for_summary(self):

        agent, model = self.agent([
            {
                "action": "respond",

                "message": (
                    "现在仅能确认你提到了"
                    "两个备选方向。"
                ),

                "focus": "decision",
            }
        ])

        _, turn = agent.reply(
            state("reflective_decision"),
            "先不要再问了，帮我总结",
        )

        self.assertEqual(
            turn.action,
            "respond",
        )

        self.assertEqual(
            turn.focus,
            "decision",
        )

        self.assertEqual(
            model.contexts[0]["program_id"],
            "reflective_decision",
        )

    # ========================================
    # 测试4：缺少教材时不能假装检索成功
    # ========================================

    def test_missing_index_is_truthfully_reported(
        self,
    ):

        agent, model = self.agent([
            {
                "action": "retrieve_knowledge",
                "query": "八木仁平 喜欢",
            },

            {
                "action": "respond",
                "message": (
                    "目前没有已接入的教材原文证据。"
                ),
            },
        ])

        _, turn = agent.reply(
            state(),
            "八木仁平的书怎么说",
        )

        self.assertFalse(
            turn.sources
        )

        self.assertIn(
            "没有",
            turn.message,
        )

        self.assertEqual(
            model.contexts[1][
                "tool_observations"
            ][0]["status"],
            "unavailable",
        )

    # ========================================
    # 测试5：检索结果进入LLM下一次决策
    # ========================================

    def test_search_loop_returns_result_to_llm(
        self,
    ):

        registry = FakeRegistry(
            lambda **kw: [
                {
                    "source": "course.pdf",
                    "quote": "原文候选",
                    "location": "p1",
                }
            ]
        )

        agent, model = self.agent(
            [
                {
                    "action": "retrieve_knowledge",
                    "query": "分析力",
                },

                {
                    "action": "ask_user",
                    "message": (
                        "你会主动分析哪些问题？"
                    ),
                    "focus": "what",
                },
            ],
            registry,
        )

        _, turn = agent.reply(
            state(),
            "查分析力课程",
        )

        self.assertEqual(
            len(registry.calls),
            1,
        )

        self.assertEqual(
            turn.sources[0]["verification"],
            "retrieved_not_checked",
        )

        self.assertEqual(
            model.contexts[1][
                "tool_observations"
            ][0]["sources"][0]["source"],
            "course.pdf",
        )

    # ========================================
    # 测试6：明显重复追问能够重试
    # ========================================

    def test_multiple_questions_retry(self):

        agent, model = self.agent([
            {
                "action": "ask_user",
                "message": (
                    "你喜欢什么？你擅长什么？"
                ),
            },

            {
                "action": "ask_user",
                "message": (
                    "你愿意主动投入什么？"
                ),
                "focus": "what",
            },
        ])

        _, turn = agent.reply(
            state(),
            "我不知道",
        )

        self.assertEqual(
            len(model.contexts),
            2,
        )

        self.assertIn(
            "重试",
            turn.trace[0],
        )

    # ========================================
    # 测试7：成长地图只读取确认笔记
    # ========================================

    def test_view_only_contains_confirmed_notes(
        self,
    ):

        view = build_growth_view([
            {
                "category": "interest",
                "content": "我自发阅读相关论文",
            },
            {
                "category": "strength",
                "content": "经常先画图分析",
            },
            {
                "category": "value",
                "content": "重视自主",
            },
            {
                "category": "unknown",
                "content": "不应显示",
            },
        ])

        self.assertEqual(
            view[0]["entries"],
            ["我自发阅读相关论文"],
        )

        self.assertEqual(
            view[1]["entries"],
            ["经常先画图分析"],
        )

        self.assertEqual(
            view[2]["entries"],
            ["重视自主"],
        )

        self.assertNotIn(
            "不应显示",
            str(view),
        )


if __name__ == "__main__":
    unittest.main()
