
import unittest

from app.agent.orchestrator import (
    AgentOrchestrator,
    AgentSession,
    ExperienceRecord,
)

from app.assessment.cross_schema import (
    CrossAssessmentDraft,
)


class NoCallRouter:

    def route(self, message, stage):
        raise AssertionError(
            "/compare不能调用自然语言Router"
        )


class NoCallKnowledgeTool:

    def answer(self, question):
        raise AssertionError(
            "/compare不能调用RAG"
        )


class FakeCrossAssessor:

    def __init__(self):
        self.calls = 0

    def assess(self, experiences, assessments=None):

        self.calls += 1

        return CrossAssessmentDraft(
            candidates=[
                {
                    "name": "有步骤的问题分析",
                    "interpretation": (
                        "两条经历中均出现逐项排查的行为。"
                    ),
                    "supports": [
                        {
                            "experience_number": 1,
                            "source_field": "action",
                            "quote": "逐项检查配置",
                        },
                        {
                            "experience_number": 2,
                            "source_field": "action",
                            "quote": "逐项测试方案",
                        },
                    ],
                    "possible_counterevidence": [],
                    "uncertainty": (
                        "尚需确认情境独立性和任务难度。"
                    ),
                }
            ],
            next_question=(
                "你还能描述一次不同情境中的经历吗？"
            ),
            limitations=(
                "目前只有两条用户自述。"
            ),
        )


class EmptyCrossAssessor:

    def assess(self, experiences, assessments=None):

        return CrossAssessmentDraft(
            candidates=[],
            next_question=(
                "是否能够提供更多不同情境的经历？"
            ),
            limitations=(
                "尚未找到符合条件的重复行为模式。"
            ),
        )


class FailingCrossAssessor:

    def assess(self, experiences, assessments=None):
        raise RuntimeError("模拟跨经历模型失败")


def two_experiences():

    return [
        ExperienceRecord(
            situation="第一次遇到技术问题。",
            action="我逐项检查配置。",
            outcome="最后解决了问题。",
        ),
        ExperienceRecord(
            situation="第二次项目遇到问题。",
            action="我逐项测试方案。",
            outcome="最后完成了任务。",
        ),
    ]


class TestCrossAgent(unittest.TestCase):

    def test_not_enough_experiences(self):

        tool = FakeCrossAssessor()

        agent = AgentOrchestrator(
            router=NoCallRouter(),
            knowledge_tool=NoCallKnowledgeTool(),
            cross_assessor=tool,
        )

        session = AgentSession(
            experiences=two_experiences()[:1],
            assessments=[None],
        )

        reply = agent.handle(
            "/compare",
            session,
        )

        self.assertEqual(
            reply.mode,
            "compare_insufficient",
        )

        self.assertEqual(
            tool.calls,
            0,
        )

    def test_compare_two_experiences(self):

        tool = FakeCrossAssessor()

        agent = AgentOrchestrator(
            router=NoCallRouter(),
            knowledge_tool=NoCallKnowledgeTool(),
            cross_assessor=tool,
        )

        session = AgentSession(
            experiences=two_experiences(),
            assessments=[None, None],
        )

        reply = agent.handle(
            "/compare",
            session,
        )

        self.assertEqual(
            reply.mode,
            "cross_assessment",
        )

        self.assertIn(
            "支持证据",
            reply.message,
        )

        self.assertIn(
            "经历1",
            reply.message,
        )

        self.assertIn(
            "经历2",
            reply.message,
        )

        self.assertEqual(tool.calls, 1)

        # 比较不能修改原始经历
        self.assertEqual(
            len(session.experiences),
            2,
        )

    def test_empty_cross_report(self):

        agent = AgentOrchestrator(
            router=NoCallRouter(),
            knowledge_tool=NoCallKnowledgeTool(),
            cross_assessor=EmptyCrossAssessor(),
        )

        session = AgentSession(
            experiences=two_experiences(),
            assessments=[None, None],
        )

        reply = agent.handle(
            "/compare",
            session,
        )

        self.assertEqual(
            reply.mode,
            "cross_assessment",
        )

        self.assertIn(
            "没有足够的跨经历证据",
            reply.message,
        )

    def test_failure_keeps_original_memory(self):

        agent = AgentOrchestrator(
            router=NoCallRouter(),
            knowledge_tool=NoCallKnowledgeTool(),
            cross_assessor=FailingCrossAssessor(),
        )

        session = AgentSession(
            experiences=two_experiences(),
            assessments=[None, None],
        )

        original_action = (
            session.experiences[0].action
        )

        reply = agent.handle(
            "/compare",
            session,
        )

        self.assertEqual(
            reply.mode,
            "compare_error",
        )

        self.assertEqual(
            len(session.experiences),
            2,
        )

        self.assertEqual(
            session.experiences[0].action,
            original_action,
        )


if __name__ == "__main__":
    unittest.main()
