
import unittest

from app.agent.guidance import (
    render_choices,
    explain_choice,
)

from app.agent.orchestrator import (
    AgentSession,
    AgentOrchestrator,
)

from app.agent.schema import RouteDecision


class NoCallRouter:
    def route(self, message, stage):
        raise AssertionError("不应该调用Router")


class FakeKnowledgeTool:
    def answer(self, question):
        raise AssertionError("不应该调用RAG")


class TestAgentGuidance(unittest.TestCase):

    def test_situation_has_options(self):
        menu = render_choices("awaiting_situation")

        self.assertIn("学习或研究", menu)
        self.assertIn("暂时想不到", menu)

    def test_action_has_options(self):
        menu = render_choices("awaiting_action")

        self.assertIn("分析和查找原因", menu)

    def test_outcome_has_options(self):
        menu = render_choices("awaiting_outcome")

        self.assertIn("没有成功", menu)

    def test_choice_does_not_change_state(self):
        session = AgentSession(
            stage="awaiting_situation"
        )

        hint = explain_choice(
            session.stage,
            "1",
        )

        self.assertIn("学习或研究", hint)

        # 选择方向并没有写入任何经历
        self.assertEqual(
            session.stage,
            "awaiting_situation",
        )

        self.assertEqual(session.draft, {})

    def test_normal_text_is_not_a_choice(self):
        hint = explain_choice(
            "awaiting_action",
            "我检查了错误日志并定位原因",
        )

        self.assertIsNone(hint)

    def test_start_command_skips_router(self):
        agent = AgentOrchestrator(
            router=NoCallRouter(),
            knowledge_tool=FakeKnowledgeTool(),
        )

        session = AgentSession()

        result = agent.handle(
            "/start",
            session,
        )

        self.assertEqual(
            session.stage,
            "awaiting_situation",
        )

        self.assertEqual(result.mode, "question")


if __name__ == "__main__":
    unittest.main()
