
import tempfile
import unittest

from pathlib import Path

from app.assessment.schema import AssessmentDraft

from app.memory.schema import (
    AgentSession,
    ExperienceRecord,
)

from app.memory.store import MemoryStore


class TestMemoryStore(unittest.TestCase):

    def setUp(self):

        self.temp = tempfile.TemporaryDirectory()

        self.path = (
            Path(self.temp.name) / "user_memory.db"
        )

        self.store = MemoryStore(self.path)

        self.key = "lesson54_demo"

    def tearDown(self):
        self.temp.cleanup()

    def sample_session(self):

        assessment = AssessmentDraft(
            observations=[
                {
                    "description": "逐项排查问题",
                    "source_field": "action",
                    "quote": "查看日志",
                }
            ],

            hypotheses=[
                {
                    "name": "问题分析",
                    "interpretation": "仍需更多经历验证",
                    "evidence_indices": [0],
                }
            ],

            follow_up_question=(
                "能否描述另一段不同情境的经历？"
            ),

            limitation=(
                "仅有单条用户自述，尚不能确认优势。"
            ),
        )

        return AgentSession(
            experiences=[
                ExperienceRecord(
                    situation="训练中出现错误",
                    action="我查看日志并排查环境",
                    outcome="训练恢复正常",
                )
            ],

            assessments=[assessment],
        )

    def test_new_session(self):

        session = self.store.load_session(
            self.key
        )

        self.assertEqual(session.stage, "idle")
        self.assertEqual(session.experiences, [])

    def test_resume_unfinished_interview(self):

        session = AgentSession(
            stage="awaiting_action",
            draft={
                "situation": "实验发生错误"
            },
        )

        self.store.save_session(
            self.key,
            session,
        )

        # 模拟重新启动程序
        recovered = MemoryStore(
            self.path
        ).load_session(self.key)

        self.assertEqual(
            recovered.stage,
            "awaiting_action",
        )

        self.assertEqual(
            recovered.draft["situation"],
            "实验发生错误",
        )

    def test_save_and_restore_assessment(self):

        self.store.save_session(
            self.key,
            self.sample_session(),
        )

        recovered = MemoryStore(
            self.path
        ).load_session(self.key)

        self.assertEqual(
            len(recovered.experiences),
            1,
        )

        self.assertEqual(
            recovered.assessments[0].hypotheses[0].name,
            "问题分析",
        )

    def test_edit_invalidates_assessment(self):

        session = self.sample_session()

        self.store.save_session(
            self.key,
            session,
        )

        updated = self.store.update_experience(
            session_id=self.key,
            session=session,
            number=1,
            field_name="action",
            new_value="我在同事指导下检查日志",
        )

        self.assertEqual(
            updated.experiences[0].action,
            "我在同事指导下检查日志",
        )

        # 更正经历后，旧分析必须失效
        self.assertIsNone(
            updated.assessments[0]
        )

        # 原始Python对象未被直接修改
        self.assertIsNotNone(
            session.assessments[0]
        )

        # 数据库里也应存储已失效的状态
        recovered = MemoryStore(
            self.path
        ).load_session(self.key)

        self.assertIsNone(
            recovered.assessments[0]
        )

    def test_delete_removes_experience_and_assessment(self):

        session = self.sample_session()

        self.store.save_session(
            self.key,
            session,
        )

        updated = self.store.delete_experience(
            self.key,
            session,
            number=1,
        )

        self.assertEqual(
            updated.experiences,
            [],
        )

        self.assertEqual(
            updated.assessments,
            [],
        )

        recovered = MemoryStore(
            self.path
        ).load_session(self.key)

        self.assertEqual(
            recovered.experiences,
            [],
        )

    def test_forget_session(self):

        self.store.save_session(
            self.key,
            self.sample_session(),
        )

        deleted = self.store.forget_session(
            self.key
        )

        self.assertTrue(deleted)

        recovered = self.store.load_session(
            self.key
        )

        self.assertEqual(
            recovered.experiences,
            [],
        )

        # 第二次删除，不应假装还有记录
        self.assertFalse(
            self.store.forget_session(self.key)
        )

    def test_sessions_are_isolated(self):

        self.store.save_session(
            "A",
            self.sample_session(),
        )

        self.store.save_session(
            "B",
            AgentSession(
                stage="awaiting_situation"
            ),
        )

        self.assertEqual(
            len(self.store.load_session("A").experiences),
            1,
        )

        self.assertEqual(
            self.store.load_session("B").stage,
            "awaiting_situation",
        )

        self.store.forget_session("B")

        # 删除B不应该影响A
        self.assertEqual(
            len(self.store.load_session("A").experiences),
            1,
        )

    def test_invalid_edit_rejected(self):

        session = self.sample_session()

        self.store.save_session(
            self.key,
            session,
        )

        with self.assertRaises(ValueError):

            self.store.update_experience(
                session_id=self.key,
                session=session,
                number=1,
                field_name="unknown",
                new_value="value",
            )

        # 无效编辑不能破坏已有分析
        recovered = self.store.load_session(
            self.key
        )

        self.assertIsNotNone(
            recovered.assessments[0]
        )


if __name__ == "__main__":
    unittest.main()
