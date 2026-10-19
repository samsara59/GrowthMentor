
"""测试Adaptive Mentor的决策、重试和安全停止。"""

import unittest

from pydantic import ValidationError

from app.agent.adaptive_mentor import (
    AdaptiveMentor,
    MentorPlanner,
    MentorState,
    Decision,
    InvalidModelResponse,
    DECISION_JSON_SCHEMA,
)

from app.domain.contracts import GrowthTask


# ============================================
# 1. 模拟导师方法
# ============================================

class FakeMethod:

    method_id = "test_method"
    name = "测试方法"
    intended_use = "供测试"
    status = "catalog_only"
    caution = "未核验"


class FakeCatalog:

    def all(self):
        return [FakeMethod()]


# ============================================
# 2. 模拟LLM
# ============================================

class FakeModel:

    def __init__(self, outputs):

        self.outputs = list(outputs)
        self.calls = []

    def decide(self, system, payload):

        self.calls.append(payload)

        answer = self.outputs.pop(0)

        if isinstance(answer, Exception):
            raise answer

        return answer


# ============================================
# 3. 模拟工具注册表
# ============================================

class FakeRegistry:

    def __init__(self):
        self.handlers = {}

    def is_bound(self, key):
        return key in self.handlers

    def bind(self, key, handler):
        self.handlers[key] = handler

    def invoke(self, key, **kwargs):
        return self.handlers[key](**kwargs)


# ============================================
# 4. 创建测试状态
# ============================================

def create_state():

    return MentorState(
        task=GrowthTask(
            task_id="t",
            program_id="self_discovery",
            goal="找到喜欢擅长的事",
        )
    )


# ============================================
# 5. 智能体测试
# ============================================

class TestAdaptiveMentor(unittest.TestCase):

    def make_agent(
        self,
        outputs,
        registry=None,
    ):

        model = FakeModel(outputs)

        agent = AdaptiveMentor(
            MentorPlanner(model),
            registry or FakeRegistry(),
            FakeCatalog(),
        )

        return agent, model

    # ----------------------------------------
    # 测试1：Schema要求action字段
    # ----------------------------------------

    def test_schema_enforces_action(self):

        self.assertIn(
            "action",
            DECISION_JSON_SCHEMA["required"],
        )

        self.assertIn(
            "ask_user",
            DECISION_JSON_SCHEMA[
                "properties"
            ]["action"]["enum"],
        )

    # ----------------------------------------
    # 测试2：正常追问
    # ----------------------------------------

    def test_single_question(self):

        agent, model = self.make_agent([
            {
                "action": "ask_user",
                "message": "你愿意做什么？",
            }
        ])

        original = create_state()

        new, turn = agent.reply(
            original,
            "想探索兴趣",
        )

        self.assertEqual(
            len(original.history),
            0,
        )

        self.assertEqual(
            len(new.history),
            2,
        )

        self.assertEqual(
            turn.action,
            "ask_user",
        )

        self.assertEqual(
            len(model.calls),
            1,
        )

    # ----------------------------------------
    # 测试3：第一次返回{}，第二次成功
    # ----------------------------------------

    def test_bad_json_then_success(self):

        agent, model = self.make_agent([
            {},

            {
                "action": "ask_user",
                "message": "最近有没有主动做的事？",
            },
        ])

        _, turn = agent.reply(
            create_state(),
            "迷茫",
        )

        self.assertEqual(
            turn.action,
            "ask_user",
        )

        self.assertEqual(
            len(model.calls),
            2,
        )

        self.assertTrue(
            any(
                "重试" in item
                for item in turn.trace
            )
        )

        self.assertIn(
            "上一轮格式错误",
            model.calls[1],
        )

    # ----------------------------------------
    # 测试4：连续三次返回{}，安全停止
    # ----------------------------------------

    def test_invalid_three_times_fails_safely(self):

        agent, model = self.make_agent([
            {},
            {},
            {},
        ])

        original = create_state()

        new, turn = agent.reply(
            original,
            "需要帮助",
        )

        self.assertEqual(
            turn.action,
            "system_error",
        )

        self.assertIn(
            "未能产生",
            turn.message,
        )

        self.assertEqual(
            len(model.calls),
            3,
        )

        self.assertEqual(
            len(original.history),
            0,
        )

        self.assertEqual(
            len(new.history),
            2,
        )

    # ----------------------------------------
    # 测试5：非法JSON文本能够重试
    # ----------------------------------------

    def test_invalid_json_text_recovers(self):

        agent, model = self.make_agent([
            InvalidModelResponse(
                "不是合法JSON"
            ),

            {
                "action": "respond",
                "message": "请谈谈你的经历",
            },
        ])

        _, turn = agent.reply(
            create_state(),
            "你好",
        )

        self.assertEqual(
            turn.action,
            "respond",
        )

        self.assertEqual(
            len(model.calls),
            2,
        )

    # ----------------------------------------
    # 测试6：检索后再次由LLM决定
    # ----------------------------------------

    def test_retrieve_then_answer(self):

        registry = FakeRegistry()

        calls = []

        def retrieve(query, limit):

            calls.append(
                (query, limit)
            )

            return [
                {
                    "source": "test.docx",
                    "quote": "测试文本",
                    "location": "一章",
                }
            ]

        registry.bind(
            "knowledge.retrieve",
            retrieve,
        )

        agent, model = self.make_agent(
            [
                {
                    "action": "retrieve_knowledge",
                    "query": "喜欢 擅长",
                },

                {
                    "action": "ask_user",
                    "message": "你想主动做什么？",
                    "method_id": "test_method",
                },
            ],

            registry,
        )

        _, turn = agent.reply(
            create_state(),
            "探索兴趣",
        )

        self.assertEqual(
            calls,
            [("喜欢 擅长", 3)],
        )

        self.assertEqual(
            turn.method_id,
            "test_method",
        )

        self.assertEqual(
            turn.sources[0]["source"],
            "test.docx",
        )

        self.assertIn(
            "test.docx",
            model.calls[1],
        )

    # ----------------------------------------
    # 测试7：缺少教材索引时不伪造检索
    # ----------------------------------------

    def test_missing_index_fallback(self):

        agent, _ = self.make_agent([
            {
                "action": "retrieve_knowledge",
                "query": "书上怎么说",
            },

            {
                "action": "respond",
                "message": "没有接入教材，不能引用",
            },
        ])

        _, turn = agent.reply(
            create_state(),
            "请查教材",
        )

        self.assertEqual(
            turn.sources,
            [],
        )

        self.assertIn(
            "不能引用",
            turn.message,
        )

    # ----------------------------------------
    # 测试8：反复要求检索时安全停止
    # ----------------------------------------

    def test_repeated_search_stop(self):

        agent, _ = self.make_agent(
            [
                {
                    "action": "retrieve_knowledge",
                    "query": "A",
                }
            ] * 3
        )

        _, turn = agent.reply(
            create_state(),
            "测试",
        )

        self.assertEqual(
            turn.action,
            "system_error",
        )

        self.assertIn(
            "安全停止",
            turn.trace[-1],
        )

    # ----------------------------------------
    # 测试9：非法动作不得污染原状态
    # ----------------------------------------

    def test_bad_model_output_recover_no_original_mutation(
        self,
    ):

        agent, _ = self.make_agent([
            {"action": "nonsense"},
            {},
            {},
        ])

        original = create_state()

        new, turn = agent.reply(
            original,
            "测试",
        )

        self.assertEqual(
            original.history,
            [],
        )

        self.assertEqual(
            turn.action,
            "system_error",
        )

        self.assertEqual(
            len(new.history),
            2,
        )

    # ----------------------------------------
    # 测试10：空输入不能进入模型
    # ----------------------------------------

    def test_empty_message_rejected(self):

        agent, _ = self.make_agent([])

        with self.assertRaises(ValueError):

            agent.reply(
                create_state(),
                " ",
            )

    # ----------------------------------------
    # 测试11：检索动作必须提供query
    # ----------------------------------------

    def test_retrieve_requires_query(self):

        with self.assertRaises(
            ValidationError
        ):

            Decision(
                action="retrieve_knowledge"
            )

    # ----------------------------------------
    # 测试12：服务故障不能伪装成正常回答
    # ----------------------------------------

    def test_transport_failure_not_silently_caught(
        self,
    ):

        agent, _ = self.make_agent([
            RuntimeError("本地服务超时")
        ])

        original = create_state()

        with self.assertRaises(
            RuntimeError
        ):

            agent.reply(
                original,
                "测试",
            )

        self.assertEqual(
            original.history,
            [],
        )


if __name__ == "__main__":
    unittest.main()
