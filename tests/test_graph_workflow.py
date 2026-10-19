
import tempfile
import unittest
from pathlib import Path

from app.agent.schema import RouteDecision

from app.agent.orchestrator import AgentOrchestrator

from app.agent.graph_workflow import build_workflow

from app.assessment.cross_schema import (
    CrossAssessmentDraft,
)

from app.assessment.verification import (
    VerificationJudgment,
)

from app.memory.schema import (
    AgentSession,
    ExperienceRecord,
)

from app.memory.store import MemoryStore

from app.rag.pipeline import RAGResponse


# ============================================
# 1. 模拟Router
# ============================================

class FakeRouter:

    def __init__(self):
        self.calls = 0

    def route(self, message, stage):
        self.calls += 1

        if "Question" in message:
            action = "knowledge"

        elif "优势" in message:
            action = "start_exploration"

        else:
            action = "other"

        return RouteDecision(
            action=action,
            reason="测试分类",
        )


# ============================================
# 2. 模拟RAG
# ============================================

class FakeRAG:

    def __init__(self):
        self.calls = []

    def answer(self, question):
        self.calls.append(question)

        return RAGResponse(
            answer="Question是指学习中的问题。",
            status="answered",
            citations=[],
        )


# ============================================
# 3. 模拟跨经历分析
# ============================================

class FakeCrossAssessor:

    def assess(
        self,
        experiences,
        assessments=None,
    ):

        return CrossAssessmentDraft.model_validate({
            "candidates": [
                {
                    "name": "问题分析",
                    "interpretation": (
                        "两条经历显示出有步骤的排查行为。"
                    ),
                    "supports": [
                        {
                            "experience_number": 1,
                            "source_field": "action",
                            "quote": "逐项检查",
                        },
                        {
                            "experience_number": 2,
                            "source_field": "action",
                            "quote": "逐项测试",
                        },
                    ],
                    "possible_counterevidence": [],
                    "uncertainty": (
                        "独立完成程度还需要验证。"
                    ),
                }
            ],
            "next_question": (
                "是否还有其他情境的类似经历？"
            ),
            "limitations": (
                "只能分析用户提供的自述材料。"
            ),
        })


# ============================================
# 4. 模拟验证模块
# ============================================

class FakeVerifier:

    def assess(
        self,
        candidate,
        question,
        answer,
        previous_turns=(),
    ):

        return VerificationJudgment(
            relation="unclear",
            quote="",
            reason=(
                "这段回答还需要更多独立行为证据。"
            ),
            next_question=(
                "你能够具体说明当时有哪些步骤"
                "是你独立完成的吗？"
            ),
        )


def example_records():
    return [
        ExperienceRecord(
            situation="训练出现错误",
            action="我逐项检查配置",
            outcome="训练恢复正常",
        ),
        ExperienceRecord(
            situation="项目出现问题",
            action="我逐项测试方案",
            outcome="项目继续推进",
        ),
    ]


# ============================================
# 5. Graph集成测试
# ============================================

class TestGraphWorkflow(unittest.TestCase):

    def setUp(self):

        self.temp = tempfile.TemporaryDirectory()

        self.memory = MemoryStore(
            Path(self.temp.name) / "memory.db"
        )

        self.router = FakeRouter()
        self.rag = FakeRAG()

        self.agent = AgentOrchestrator(
            router=self.router,
            knowledge_tool=self.rag,
            assessor=None,
            cross_assessor=FakeCrossAssessor(),
            verifier=FakeVerifier(),
        )

        self.graph = build_workflow(
            agent=self.agent,
            memory=self.memory,
        )

    def tearDown(self):
        self.temp.cleanup()

    def ask(self, message, session_id="demo"):
        return self.graph.invoke({
            "message": message,
            "session_id": session_id,
        })

    # ========================================
    # 测试1：Supervisor正确调用RAG
    # ========================================

    def test_rag_routing(self):

        result = self.ask(
            "Question是什么意思？"
        )

        self.assertEqual(
            result["reply"].mode,
            "knowledge",
        )

        self.assertIn(
            "rag_agent",
            result["path"],
        )

        self.assertEqual(
            self.router.calls,
            1,
        )

        self.assertEqual(
            self.rag.calls,
            ["Question是什么意思？"],
        )

    # ========================================
    # 测试2：跨轮访谈与SQLite保存
    # ========================================

    def test_interview_persistence(self):

        self.ask("/start")

        self.ask(
            "上周我训练模型时发生了环境错误。"
        )

        session = self.memory.load_session("demo")

        self.assertEqual(
            session.stage,
            "awaiting_action",
        )

        self.ask(
            "我查看日志并逐项检查了环境配置。"
        )

        result = self.ask(
            "最后找到了原因，模型恢复正常训练。"
        )

        self.assertEqual(
            result["reply"].mode,
            "recorded",
        )

        # 新建图对象模拟程序重启
        new_graph = build_workflow(
            agent=self.agent,
            memory=MemoryStore(self.memory.path),
        )

        restored = new_graph.invoke({
            "message": "/status",
            "session_id": "demo",
        })

        self.assertIn(
            "已完成经历：1",
            restored["reply"].message,
        )

        # 正在访谈时不会反复调用Router
        self.assertEqual(
            self.router.calls,
            0,
        )

    # ========================================
    # 测试3：跨经历分析与动态验证
    # ========================================

    def test_compare_and_verification(self):

        self.memory.save_session(
            "demo",
            AgentSession(
                experiences=example_records(),
                assessments=[None, None],
            ),
        )

        compared = self.ask("/compare")

        self.assertEqual(
            compared["reply"].mode,
            "cross_assessment",
        )

        self.assertIn(
            "assessment_agent",
            compared["path"],
        )

        started = self.ask("/verify 1")

        self.assertEqual(
            started["reply"].mode,
            "verification_question",
        )

        answered = self.ask(
            "我当时独立检查了多个方案并记录结果。"
        )

        self.assertEqual(
            answered["reply"].mode,
            "verification_question",
        )

        restored = self.memory.load_session("demo")

        self.assertEqual(
            restored.stage,
            "awaiting_verification",
        )

        self.assertEqual(
            len(restored.verification.turns),
            1,
        )

    # ========================================
    # 测试4：删除必须显式确认
    # ========================================

    def test_delete_requires_confirmation(self):

        self.memory.save_session(
            "demo",
            AgentSession(
                experiences=example_records(),
                assessments=[None, None],
            ),
        )

        self.ask("/delete 1")

        self.assertEqual(
            len(
                self.memory.load_session(
                    "demo"
                ).experiences
            ),
            2,
        )

        self.ask("/delete 1 YES")

        self.assertEqual(
            len(
                self.memory.load_session(
                    "demo"
                ).experiences
            ),
            1,
        )

    # ========================================
    # 测试5：删除会话后不能立即重建
    # ========================================

    def test_forget_does_not_recreate_memory(self):

        self.ask("/start")

        self.ask("/forget")

        # 未确认时数据库仍保留记录
        self.assertEqual(
            self.memory.load_session("demo").stage,
            "awaiting_situation",
        )

        result = self.ask("/forget YES")

        self.assertTrue(
            result["skip_save"]
        )

        # 直接查询数据库表，确认记录已删除
        import sqlite3
        from contextlib import closing

        with closing(
            sqlite3.connect(self.memory.path)
        ) as conn:

            count = conn.execute(
                "SELECT COUNT(*) FROM agent_sessions "
                "WHERE session_id = ?",
                ("demo",),
            ).fetchone()[0]

        self.assertEqual(count, 0)

    # ========================================
    # 测试6：图结构可以导出
    # ========================================

    def test_draw_mermaid(self):

        diagram = (
            self.graph.get_graph().draw_mermaid()
        )

        self.assertIn(
            "supervisor",
            diagram,
        )

        self.assertIn(
            "assessment_agent",
            diagram,
        )

        self.assertIn(
            "memory_save",
            diagram,
        )


if __name__ == "__main__":
    unittest.main()
