
import unittest

from app.agent.schema import RouteDecision

from app.agent.orchestrator import (
    AgentOrchestrator,
    AgentSession,
)

from app.assessment.schema import AssessmentDraft


class FakeRouter:
    """模拟每轮的任务路由。"""

    def __init__(self):

        self.actions = iter([
            "start_exploration",
            "experience_reply",
            "experience_reply",
            "experience_reply",
        ])

    def route(self, message, stage):

        return RouteDecision(
            action=next(self.actions),
            reason="测试路由",
        )


class FakeAssessor:
    """模拟成功的经历分析。"""

    def assess(self, experience):

        return AssessmentDraft(
            observations=[
                {
                    "description": "逐项排查技术问题",
                    "source_field": "action",
                    "quote": "检查错误日志",
                }
            ],

            hypotheses=[
                {
                    "name": "问题分析",
                    "interpretation": (
                        "需要其他经历进一步验证。"
                    ),
                    "evidence_indices": [0],
                }
            ],

            follow_up_question=(
                "你还有其他独立解决问题的经历吗？"
            ),

            limitation=(
                "只有一次用户自述，不能确定稳定优势。"
            ),
        )


class FailingAssessor:
    """模拟Assessment调用失败。"""

    def assess(self, experience):
        raise RuntimeError("模拟模型失败")


class FakeKnowledgeTool:
    """本测试不应该调用RAG。"""

    def answer(self, question):
        raise AssertionError(
            "本测试不应该调用RAG"
        )


def complete_interview(
    agent: AgentOrchestrator,
    session: AgentSession,
):
    """模拟完成一次三阶段访谈。"""

    # 启动
    agent.handle(
        "开始优势探索",
        session,
    )

    # 情境
    agent.handle(
        "上周我的模型训练一直报错，影响了项目进度。",
        session,
    )

    # 行动
    agent.handle(
        "我检查错误日志，并且逐项排查依赖问题。",
        session,
    )

    # 结果
    return agent.handle(
        "最终修复了环境问题，模型恢复训练。",
        session,
    )


class TestAgentAssessment(unittest.TestCase):

    def test_assessment_after_interview(self):

        agent = AgentOrchestrator(
            router=FakeRouter(),
            knowledge_tool=FakeKnowledgeTool(),
            assessor=FakeAssessor(),
        )

        session = AgentSession()

        reply = complete_interview(
            agent,
            session,
        )

        # Agent应当自动执行Assessment
        self.assertEqual(
            reply.mode,
            "assessment",
        )

        # 回复应包含候选方向
        self.assertIn(
            "问题分析",
            reply.message,
        )

        # 回复应包含下一轮建议问题
        self.assertIn(
            "下一轮建议问题",
            reply.message,
        )

        # 经历应成功保存
        self.assertEqual(
            len(session.experiences),
            1,
        )

        # 收集结束后恢复idle
        self.assertEqual(
            session.stage,
            "idle",
        )

    def test_assessment_failure_preserves_experience(self):

        agent = AgentOrchestrator(
            router=FakeRouter(),
            knowledge_tool=FakeKnowledgeTool(),
            assessor=FailingAssessor(),
        )

        session = AgentSession()

        reply = complete_interview(
            agent,
            session,
        )

        # 应记录分析失败
        self.assertEqual(
            reply.mode,
            "assessment_error",
        )

        # 但经历不能丢失
        self.assertEqual(
            len(session.experiences),
            1,
        )

        self.assertEqual(
            session.stage,
            "idle",
        )


if __name__ == "__main__":
    unittest.main()
