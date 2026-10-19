
import unittest

from app.agent.orchestrator import (
    AgentOrchestrator,
    AgentSession,
)

from app.rag.pipeline import RAGResponse


class NoCallRouter:
    """访谈期间不允许调用Router。"""

    def route(self, message, stage):
        raise AssertionError(
            "进行中的访谈不应该调用LLM Router"
        )


class FakeKnowledgeTool:
    def answer(self, question):
        return RAGResponse(
            answer="这是教材回答。",
            status="answered",
            citations=[],
        )


class TestAgentStageRouting(unittest.TestCase):

    def setUp(self):
        self.agent = AgentOrchestrator(
            router=NoCallRouter(),
            knowledge_tool=FakeKnowledgeTool(),
            assessor=None,
        )

    def test_situation_directly_advances(self):
        session = AgentSession(
            stage="awaiting_situation"
        )

        self.agent.handle(
            "上周我训练模型时出现了环境错误。",
            session,
        )

        self.assertEqual(
            session.stage,
            "awaiting_action",
        )

        self.assertIn(
            "situation",
            session.draft,
        )

    def test_action_directly_advances(self):
        session = AgentSession(
            stage="awaiting_action",
            draft={
                "situation": "上周训练出现环境错误。"
            },
        )

        self.agent.handle(
            "我检查日志并逐项排查依赖问题。",
            session,
        )

        self.assertEqual(
            session.stage,
            "awaiting_outcome",
        )

        self.assertIn(
            "action",
            session.draft,
        )

    def test_complete_experience(self):
        session = AgentSession(
            stage="awaiting_situation"
        )

        self.agent.handle(
            "上周训练出现了环境配置错误。",
            session,
        )

        self.agent.handle(
            "我检查日志并排查了环境依赖问题。",
            session,
        )

        reply = self.agent.handle(
            "最后解决了环境错误并恢复训练。",
            session,
        )

        self.assertEqual(session.stage, "idle")
        self.assertEqual(len(session.experiences), 1)
        self.assertEqual(len(session.assessments), 1)
        self.assertEqual(reply.mode, "recorded")

    def test_ask_command_during_interview(self):
        session = AgentSession(
            stage="awaiting_action",
            draft={
                "situation": "发生了一件困难的事情。"
            },
        )

        reply = self.agent.handle(
            "/ask Question是什么意思？",
            session,
        )

        self.assertEqual(reply.mode, "knowledge")

        # 临时询问教材不会破坏访谈阶段
        self.assertEqual(
            session.stage,
            "awaiting_action",
        )

    def test_start_cannot_erase_draft(self):
        session = AgentSession(
            stage="awaiting_action",
            draft={
                "situation": "这是一段已经记录的真实情境。"
            },
        )

        self.agent.handle("/start", session)

        self.assertEqual(
            session.stage,
            "awaiting_action",
        )

        self.assertIn(
            "situation",
            session.draft,
        )


if __name__ == "__main__":
    unittest.main()
