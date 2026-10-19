
import json
import sqlite3
import tempfile
import unittest

from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from app.agent.orchestrator import (
    AgentOrchestrator,
    AgentSession,
    ExperienceRecord,
)

from app.assessment.cross_schema import (
    CrossAssessmentDraft,
)

from app.assessment.verification import (
    EvidenceVerifier,
    VerificationJudgment,
)

from app.memory.store import MemoryStore


# ============================================
# 1. 模拟Router和RAG
# ============================================

class NoCall:
    """验证命令不应该调用Router或RAG。"""

    def route(self, message, stage):
        raise AssertionError(
            "不应该调用Router"
        )

    def answer(self, question):
        raise AssertionError(
            "不应该调用RAG"
        )


# ============================================
# 2. 模拟跨经历分析模块
# ============================================

class FakeCross:

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
                        "两条经历均有逐项检查过程。"
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
                            "quote": "逐项验证方案",
                        },
                    ],
                    "possible_counterevidence": [],
                    "uncertainty": (
                        "独立性和其他情境仍待验证。"
                    ),
                }
            ],
            "next_question": (
                "能否提供不同情境中的行为？"
            ),
            "limitations": (
                "用户自述不能独立证实稳定优势。"
            ),
        })


# ============================================
# 3. 模拟动态验证模型
# ============================================

class FakeVerifier:

    def __init__(self):
        self.calls = 0

    def assess(
        self,
        candidate,
        question,
        answer,
        previous_turns=(),
    ):

        self.calls += 1

        return VerificationJudgment(
            relation="unclear",
            quote="",
            reason=(
                "当前回答缺少足够的具体证据。"
            ),
            next_question=(
                f"请进一步说明第{self.calls}轮"
                "涉及的具体行动和结果。"
            ),
        )


class FailingVerifier:

    def assess(self, **kwargs):
        raise RuntimeError(
            "模拟推理失败"
        )


class InvalidQuoteClient:
    """模拟引用了不存在的用户原话。"""

    def generate(self, prompt):

        return json.dumps(
            {
                "relation": "supports",
                "quote": "用户没有说过这句话",
                "reason": (
                    "可能存在支持证据。"
                ),
                "next_question": (
                    "请描述另一个具体情境。"
                ),
            },
            ensure_ascii=False,
        )


# ============================================
# 4. 构造测试用的历史经历
# ============================================

def sample_experiences():

    return [
        ExperienceRecord(
            situation="训练出现问题",
            action="我逐项检查配置并恢复训练",
            outcome="模型恢复正常",
        ),
        ExperienceRecord(
            situation="研究遇到问题",
            action="我逐项验证方案并梳理公式",
            outcome="能够解释主要推导",
        ),
    ]


# ============================================
# 5. 验证工作流测试
# ============================================

class TestVerificationWorkflow(unittest.TestCase):

    def build_agent(
        self,
        verifier=None,
    ):

        return AgentOrchestrator(
            router=NoCall(),
            knowledge_tool=NoCall(),
            cross_assessor=FakeCross(),
            verifier=(
                verifier
                if verifier is not None
                else FakeVerifier()
            ),
        )

    def make_session(
        self,
        agent,
    ):

        session = AgentSession(
            experiences=sample_experiences(),
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

        self.assertIsNotNone(
            session.cross_report,
        )

        return session

    # ========================================
    # 测试1：三轮动态验证
    # ========================================

    def test_start_and_three_adaptive_rounds(self):

        agent = self.build_agent()
        session = self.make_session(agent)

        reply = agent.handle(
            "/verify 1",
            session,
        )

        self.assertEqual(
            reply.mode,
            "verification_question",
        )

        self.assertEqual(
            session.stage,
            "awaiting_verification",
        )

        questions = []

        for i in range(3):

            questions.append(
                session.verification.pending_question
            )

            reply = agent.handle(
                f"我正在描述第{i + 1}轮"
                "遇到的具体问题及实际行动。",
                session,
            )

            self.assertIn(
                "原文",
                reply.message,
            )

        self.assertEqual(
            session.stage,
            "idle",
        )

        self.assertEqual(
            len(session.verification.turns),
            3,
        )

        self.assertTrue(
            session.verification.completed
        )

        # 验证第二、三轮问题不是固定重复的
        self.assertNotEqual(
            questions[1],
            questions[2],
        )

    # ========================================
    # 测试2：没有跨经历报告不能启动验证
    # ========================================

    def test_verification_requires_report(self):

        agent = self.build_agent()

        session = AgentSession(
            experiences=sample_experiences(),
            assessments=[None, None],
        )

        reply = agent.handle(
            "/verify 1",
            session,
        )

        self.assertEqual(
            reply.mode,
            "verification_unavailable",
        )

    # ========================================
    # 测试3：拒绝模型编造的原文引用
    # ========================================

    def test_invalid_quote_is_rejected(self):

        verifier = EvidenceVerifier(
            client=InvalidQuoteClient(),
            max_attempts=1,
        )

        agent = self.build_agent()
        session = self.make_session(agent)

        candidate = (
            session.cross_report.candidates[0]
        )

        with self.assertRaises(ValueError):

            verifier.assess(
                candidate=candidate,
                question="请举一个例子。",
                answer=(
                    "我尝试过拆解问题并检查细节。"
                ),
            )

    # ========================================
    # 测试4：模型失败后保留用户回答
    # ========================================

    def test_failed_review_preserves_reply_and_retry(self):

        agent = self.build_agent(
            verifier=FailingVerifier()
        )

        session = self.make_session(agent)

        agent.handle(
            "/verify 1",
            session,
        )

        answer = (
            "这是我最近一次遇到困难的"
            "具体情况和实际行动。"
        )

        reply = agent.handle(
            answer,
            session,
        )

        self.assertEqual(
            reply.mode,
            "verification_error",
        )

        self.assertEqual(
            session.verification.pending_answer,
            answer,
        )

        retry = agent.handle(
            "/retryverify",
            session,
        )

        self.assertEqual(
            retry.mode,
            "verification_error",
        )

        self.assertEqual(
            len(session.verification.turns),
            0,
        )

    # ========================================
    # 测试5：持久化验证状态与读取旧版快照
    # ========================================

    def test_memory_persists_verification_and_legacy_v1(
        self,
    ):

        agent = self.build_agent()
        session = self.make_session(agent)

        agent.handle(
            "/verify 1",
            session,
        )

        agent.handle(
            "我在其他问题中也检查过多个可行方案。",
            session,
        )

        # 使用临时数据库，不影响用户真实记忆
        with tempfile.TemporaryDirectory() as temp:

            db_path = Path(temp) / "memory.db"

            store = MemoryStore(
                db_path
            )

            # A. 保存新版会话
            store.save_session(
                "new",
                session,
            )

            # B. 创建一个新的Store模拟程序重启
            restored = MemoryStore(
                db_path
            ).load_session("new")

            self.assertEqual(
                restored.stage,
                "awaiting_verification",
            )

            self.assertEqual(
                len(restored.verification.turns),
                1,
            )

            self.assertIsNotNone(
                restored.cross_report,
            )

            # C. 构造第54课的旧版version=1快照
            old_snapshot = {
                "version": 1,
                "stage": "idle",
                "draft": {},
                "experiences": [
                    asdict(item)
                    for item in sample_experiences()
                ],
                "assessments": [
                    None,
                    None,
                ],
            }

            old_json = json.dumps(
                old_snapshot,
                ensure_ascii=False,
            )

            # D. 关键修复：
            # closing()负责关闭SQLite连接
            # with conn负责事务提交或回滚
            with closing(
                sqlite3.connect(db_path)
            ) as conn:

                with conn:
                    conn.execute(
                        """
                        INSERT INTO agent_sessions (
                            session_id,
                            snapshot_json
                        )
                        VALUES (?, ?)
                        """,
                        (
                            "old",
                            old_json,
                        ),
                    )

            # 到这里，SQLite连接已经关闭

            # E. 验证可以读取旧版本快照
            legacy = store.load_session(
                "old"
            )

            self.assertEqual(
                len(legacy.experiences),
                2,
            )

            self.assertIsNone(
                legacy.cross_report
            )

        # TemporaryDirectory在这里删除memory.db
        # 因为连接已关闭，Windows不应再出现WinError 32

    # ========================================
    # 测试6：更正经历后使旧验证失效
    # ========================================

    def test_edit_invalidates_cached_verification(self):

        agent = self.build_agent()
        session = self.make_session(agent)

        agent.handle(
            "/verify 1",
            session,
        )

        with tempfile.TemporaryDirectory() as temp:

            store = MemoryStore(
                Path(temp) / "memory.db"
            )

            store.save_session(
                "x",
                session,
            )

            edited = store.update_experience(
                session_id="x",
                session=session,
                number=1,
                field_name="action",
                new_value=(
                    "我在指导下排查并解决了问题。"
                ),
            )

            self.assertIsNone(
                edited.cross_report
            )

            self.assertIsNone(
                edited.verification
            )

            self.assertEqual(
                edited.stage,
                "idle",
            )

            recovered = store.load_session(
                "x"
            )

            self.assertIsNone(
                recovered.cross_report
            )

            self.assertIsNone(
                recovered.verification
            )


if __name__ == "__main__":
    unittest.main()
