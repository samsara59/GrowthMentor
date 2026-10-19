
import unittest

from app.agent.schema import RouteDecision

from app.agent.orchestrator import (
    AgentOrchestrator,
    AgentSession,
)

from app.rag.pipeline import RAGResponse


class FakeRouter:

    def __init__(self, actions):
        self.actions = iter(actions)

    def route(self, message, stage):
        return RouteDecision(
            action=next(self.actions),
            reason="测试路由",
        )


class FakeKnowledgeTool:

    def __init__(self):
        self.calls = []

    def answer(self, question):
        self.calls.append(question)

        return RAGResponse(
            answer="Question表示学习中的问题。",
            status="answered",
            citations=[],
        )


class TestAgentOrchestrator(unittest.TestCase):

    def test_knowledge_route(self):
        tool = FakeKnowledgeTool()

        agent = AgentOrchestrator(
            router=FakeRouter(["knowledge"]),
            knowledge_tool=tool,
        )

        reply = agent.handle(
            "Question是什么意思？",
            AgentSession(),
        )

        self.assertEqual(reply.mode, "knowledge")
        self.assertEqual(len(tool.calls), 1)

    def test_start_exploration(self):
        agent = AgentOrchestrator(
            router=FakeRouter(["start_exploration"]),
            knowledge_tool=FakeKnowledgeTool(),
        )

        session = AgentSession()

        reply = agent.handle(
            "我想知道自己的优势",
            session,
        )

        self.assertEqual(
            session.stage,
            "awaiting_situation",
        )

        self.assertEqual(reply.mode, "question")

    def test_collect_complete_experience(self):
        agent = AgentOrchestrator(
            router=FakeRouter([
                "start_exploration",
                "experience_reply",
                "experience_reply",
                "experience_reply",
            ]),
            knowledge_tool=FakeKnowledgeTool(),
        )

        session = AgentSession()

        agent.handle("开始优势探索", session)

        agent.handle(
            "上周模型训练一直报错，影响了实验进度。",
            session,
        )

        self.assertEqual(
            session.stage,
            "awaiting_action",
        )

        agent.handle(
            "我检查日志并逐步定位了环境配置问题。",
            session,
        )

        self.assertEqual(
            session.stage,
            "awaiting_outcome",
        )

        reply = agent.handle(
            "最后成功恢复了训练并完成实验。",
            session,
        )

        self.assertEqual(session.stage, "idle")
        self.assertEqual(len(session.experiences), 1)
        self.assertEqual(reply.mode, "recorded")

        self.assertIn(
            "仅凭一次经历",
            reply.message,
        )

    def test_knowledge_does_not_erase_session(self):
        agent = AgentOrchestrator(
            router=FakeRouter([
                "start_exploration",
                "knowledge",
            ]),
            knowledge_tool=FakeKnowledgeTool(),
        )

        session = AgentSession()

        agent.handle("开始优势探索", session)

        agent.handle("什么是学习力？", session)

        self.assertEqual(
            session.stage,
            "awaiting_situation",
        )

    def test_unsupported_request(self):
        tool = FakeKnowledgeTool()

        agent = AgentOrchestrator(
            router=FakeRouter(["other"]),
            knowledge_tool=tool,
        )

        reply = agent.handle(
            "明天天气怎么样？",
            AgentSession(),
        )

        self.assertEqual(reply.mode, "unsupported")
        self.assertEqual(tool.calls, [])

    def test_unsolicited_experience_reply(self):
        agent = AgentOrchestrator(
            router=FakeRouter(["experience_reply"]),
            knowledge_tool=FakeKnowledgeTool(),
        )

        session = AgentSession()

        reply = agent.handle(
            "上周我完成了一个重要的研究项目。",
            session,
        )

        self.assertEqual(
            len(session.experiences),
            0,
        )

        self.assertEqual(reply.mode, "question")


if __name__ == "__main__":
    unittest.main()
