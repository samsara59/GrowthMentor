
import json
import tempfile
import unittest

from pathlib import Path

from app.assessment.assessor import ExperienceAssessor
from app.assessment.schema import AssessmentDraft

from app.agent.orchestrator import AgentOrchestrator

from app.memory.schema import (
    AgentSession,
    ExperienceRecord,
)

from app.memory.store import MemoryStore


class SequenceClient:
    """模拟LLM依次返回不同的JSON。"""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return json.dumps(
            next(self.responses),
            ensure_ascii=False,
        )


class StaticAssessor:
    """模拟一次成功的Assessment。"""

    def assess(self, experience):
        return AssessmentDraft(
            observations=[
                {
                    "description": "检查技术问题",
                    "source_field": "action",
                    "quote": "检查错误日志",
                }
            ],
            hypotheses=[
                {
                    "name": "问题分析",
                    "interpretation": "仍需进一步验证。",
                    "evidence_indices": [0],
                }
            ],
            follow_up_question=(
                "在其他情境中也采取过类似行动吗？"
            ),
            limitation=(
                "当前只有一次用户自述。"
            ),
        )


class FailedAssessor:
    def assess(self, experience):
        raise RuntimeError("模拟模型调用失败")


class NoCallRouter:
    def route(self, message, stage):
        raise AssertionError(
            "重新分析不应该调用Router"
        )


class NoCallRAG:
    def answer(self, question):
        raise AssertionError(
            "重新分析不应该调用RAG"
        )


def sample_experience():
    return ExperienceRecord(
        situation="上周训练模型时出现问题。",
        action="我检查错误日志并排查依赖。",
        outcome="模型最终恢复了正常训练。",
    )


class TestAssessmentRecovery(unittest.TestCase):

    def test_invalid_quote_can_be_corrected(self):

        invalid = {
            "observations": [
                {
                    "description": "分析问题",
                    "source_field": "action",
                    "quote": "我领导了整个研究团队",
                }
            ],
            "hypotheses": [],
            "follow_up_question": (
                "是否还有其他解决问题的经历？"
            ),
            "limitation": "只有一次经历。",
        }

        valid = {
            "observations": [
                {
                    "description": "检查错误原因",
                    "source_field": "action",
                    "quote": "检查错误日志",
                }
            ],
            "hypotheses": [
                {
                    "name": "问题分析",
                    "interpretation": "需要更多证据验证。",
                    "evidence_indices": [0],
                }
            ],
            "follow_up_question": (
                "其他任务也采用过类似方法吗？"
            ),
            "limitation": "只有一次用户自述。",
        }

        client = SequenceClient([
            invalid,
            valid,
        ])

        assessor = ExperienceAssessor(
            client=client,
            max_attempts=2,
        )

        result = assessor.assess(
            sample_experience()
        )

        self.assertEqual(client.calls, 2)

        self.assertEqual(
            result.observations[0].quote,
            "检查错误日志",
        )

    def test_repeated_invalid_quote_is_rejected(self):

        invalid = {
            "observations": [
                {
                    "description": "带领大型团队",
                    "source_field": "action",
                    "quote": "我领导了整个研究团队",
                }
            ],
            "hypotheses": [],
            "follow_up_question": (
                "你有没有其他研究经历？"
            ),
            "limitation": "只有一次自述经历。",
        }

        client = SequenceClient([
            invalid,
            invalid,
        ])

        assessor = ExperienceAssessor(
            client,
            max_attempts=2,
        )

        with self.assertRaises(ValueError):
            assessor.assess(
                sample_experience()
            )

        self.assertEqual(client.calls, 2)

    def test_reanalyze_updates_saved_session(self):

        session = AgentSession(
            experiences=[
                sample_experience()
            ],
            assessments=[None],
        )

        agent = AgentOrchestrator(
            router=NoCallRouter(),
            knowledge_tool=NoCallRAG(),
            assessor=StaticAssessor(),
        )

        reply = agent.reanalyze_experience(
            number=1,
            session=session,
        )

        self.assertEqual(
            reply.mode,
            "assessment",
        )

        self.assertIsNotNone(
            session.assessments[0]
        )

        with tempfile.TemporaryDirectory() as temp:

            db = Path(temp) / "memory.db"

            store = MemoryStore(db)

            store.save_session(
                "demo",
                session,
            )

            recovered = store.load_session(
                "demo"
            )

            self.assertEqual(
                recovered.assessments[0]
                .hypotheses[0].name,
                "问题分析",
            )

    def test_failure_preserves_original_experience(self):

        session = AgentSession(
            experiences=[
                sample_experience()
            ],
            assessments=[None],
        )

        agent = AgentOrchestrator(
            router=NoCallRouter(),
            knowledge_tool=NoCallRAG(),
            assessor=FailedAssessor(),
        )

        reply = agent.reanalyze_experience(
            number=1,
            session=session,
        )

        self.assertEqual(
            reply.mode,
            "assessment_error",
        )

        self.assertEqual(
            len(session.experiences),
            1,
        )

        self.assertIsNone(
            session.assessments[0]
        )

        self.assertIn(
            "RuntimeError",
            reply.message,
        )


if __name__ == "__main__":
    unittest.main()
