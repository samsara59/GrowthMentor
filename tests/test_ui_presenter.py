
import unittest

from app.ui.presenter import (
    validate_session_id,
    format_trace,
    build_graphviz_dot,
    stage_description,
    get_summary,
)

from app.memory.schema import (
    AgentSession,
    ExperienceRecord,
    VerificationSession,
)


class TestUIPresenter(unittest.TestCase):

    def test_session_id_validation(self):

        self.assertEqual(
            validate_session_id("lesson54_demo"),
            "lesson54_demo",
        )

        with self.assertRaises(ValueError):
            validate_session_id("../private")

        with self.assertRaises(ValueError):
            validate_session_id("")

    def test_execution_trace(self):

        path = [
            "memory_load",
            "supervisor",
            "assessment_agent",
            "memory_save",
        ]

        result = format_trace(path)

        self.assertIn(
            "Supervisor",
            result,
        )

        self.assertIn(
            "Assessment Agent",
            result,
        )

        self.assertNotIn(
            "RAG Agent",
            result,
        )

    def test_graph_contains_agent_nodes(self):

        dot = build_graphviz_dot()

        for node in [
            "supervisor",
            "rag_agent",
            "interview_agent",
            "assessment_agent",
            "verification_agent",
            "memory_agent",
            "memory_save",
        ]:

            self.assertIn(
                f'"{node}"',
                dot,
            )

    def test_executed_nodes_are_highlighted(self):

        path = [
            "memory_load",
            "supervisor",
            "rag_agent",
            "memory_save",
        ]

        dot = build_graphviz_dot(path)

        # 已执行的RAG节点被标记
        self.assertIn(
            '"rag_agent" [label="RAG Agent", '
            'fillcolor="#DFF3E7"',
            dot,
        )

        # 未执行的Assessment节点保持普通状态
        self.assertIn(
            '"assessment_agent" '
            '[label="Assessment Agent", '
            'fillcolor="#F1F3F6"',
            dot,
        )

    def test_stage_guidance(self):

        session = AgentSession(
            stage="awaiting_action",
            draft={
                "situation": "训练出现错误"
            },
        )

        message = stage_description(session)

        self.assertIn(
            "行动",
            message,
        )

    def test_verification_resume_guidance(self):

        session = AgentSession(
            stage="awaiting_verification",
            verification=VerificationSession(
                candidate_number=1,
                candidate_name="问题分析",
                source_fingerprint="test",
                pending_question=(
                    "你当时具体做了什么？"
                ),
            ),
        )

        self.assertIn(
            "你当时具体做了什么",
            stage_description(session),
        )

        session.verification.pending_answer = (
            "我当时先检查了错误日志。"
        )

        self.assertIn(
            "/retryverify",
            stage_description(session),
        )

    def test_summary(self):

        session = AgentSession(
            experiences=[
                ExperienceRecord(
                    situation="训练出现错误",
                    action="检查日志",
                    outcome="恢复训练",
                )
            ],
            assessments=[None],
        )

        result = get_summary(session)

        self.assertEqual(
            result["experiences"],
            1,
        )

        self.assertEqual(
            result["assessments"],
            0,
        )

        self.assertEqual(
            result["candidates"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
