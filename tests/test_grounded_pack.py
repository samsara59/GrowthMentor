
"""第72课：真实原文匹配、方法目录与证据编号测试。"""

import json
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace

from app.knowledge.source_provenance import (
    SourceCatalog,
)

from app.methods.grounded_pack import (
    GroundedMethodCoach,
)


# 仅供单元测试使用的虚构材料
QUOTE = (
    "主动解决一个没有人布置的问题，"
    "需要区分自发投入与任务完成。"
) * 3


# ============================================
# 1. 模拟现有MethodCatalog
# ============================================

class FakeMethod:

    def __init__(self, key):

        self.method_id = key
        self.name = key

        self.intended_use = (
            "检验用户目标与当前信息"
        )

        self.caution = (
            "方法尚未核实"
        )

        self.status = "catalog_only"


class FakeMethods:

    def all(self):

        return [
            FakeMethod(key)

            for key in (
                "yagi_self_understanding",
                "local_strengths_courses",
                "contradiction_lens",
            )
        ]


# ============================================
# 2. 模拟教材检索
# ============================================

class FakeSearch:

    def __init__(
        self,
        excerpt=QUOTE,
    ):

        self.excerpt = excerpt
        self.calls = []

    def search(
        self,
        question,
        *,
        source=None,
        limit=5,
    ):

        self.calls.append(
            (question, source, limit)
        )

        return [
            SimpleNamespace(
                id="E1",
                source=source,
                location="record=1",
                excerpt=self.excerpt,
            )
        ]


# ============================================
# 3. 模拟大语言模型
# ============================================

class FakeModel:

    def __init__(
        self,
        answer,
    ):

        self.answer = answer
        self.calls = []

    def generate_json(
        self,
        system,
        payload,
        schema,
    ):

        self.calls.append(
            json.loads(payload)
        )

        return self.answer


# ============================================
# 4. 模拟模型回答
# ============================================

def output(
    status="limited_application",
    ids=None,
):

    if ids is None:

        ids = (
            ["E1"]
            if status == "limited_application"
            else []
        )

    return {
        "status": status,

        "reading_of_source": (
            "材料描述了对某种问题的"
            "自发投入线索，"
            "需要保留不确定性。"
        ),

        "contextual_application": (
            "可以回顾一个没有外部任务要求时"
            "主动研究的具体问题。"
        ),

        "one_question": (
            "你最近主动研究了什么问题？"
        ),

        "limitations": (
            "这只是经字符匹配的片段，"
            "不代表作者完整方法已经核实。"
        ),

        "evidence_ids": ids,
    }


# ============================================
# 5. 六项测试
# ============================================

class TestGroundedPack(unittest.TestCase):

    def setUp(self):

        self.temp = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.temp.name
        )

        (
            self.root / "course.txt"
        ).write_text(
            QUOTE,
            encoding="utf-8",
        )

        self.catalog = SourceCatalog(
            self.root
        )

    def tearDown(self):

        self.temp.cleanup()

    # ========================================
    # 测试1：复用现有方法目录
    # ========================================

    def test_uses_existing_method_catalog(self):

        agent = GroundedMethodCoach(
            FakeSearch(),
            self.catalog,
            FakeModel(output()),
            FakeMethods(),
        )

        self.assertIn(
            "yagi_self_understanding",
            agent.methods,
        )

        self.assertIn(
            "contradiction_lens",
            agent.methods,
        )

    # ========================================
    # 测试2：只有匹配原文的证据进入LLM
    # ========================================

    def test_verified_excerpt_reaches_model(self):

        search = FakeSearch()

        model = FakeModel(
            output()
        )

        run = GroundedMethodCoach(
            search,
            self.catalog,
            model,
            FakeMethods(),
        ).run(
            "yagi_self_understanding",
            "course.txt",
            "自发兴趣与完成任务有什么区别？",
        )

        self.assertEqual(
            len(run.verified),
            1,
        )

        self.assertEqual(
            run.verified[0].match_status,
            "matched_original_text",
        )

        self.assertEqual(
            run.draft.evidence_ids,
            ["E1"],
        )

        self.assertEqual(
            len(model.calls),
            1,
        )

        self.assertIn(
            "text_matched_excerpts_not_semantically_verified",
            model.calls[0],
        )

        self.assertEqual(
            search.calls[0][1],
            "course.txt",
        )

    # ========================================
    # 测试3：不匹配原文时不调用LLM
    # ========================================

    def test_unmatched_excerpt_never_calls_model(self):

        model = FakeModel(
            output()
        )

        run = GroundedMethodCoach(
            FakeSearch(
                "不属于当前教材中的其他虚构文字"
                * 5
            ),
            self.catalog,
            model,
            FakeMethods(),
        ).run(
            "local_strengths_courses",
            "course.txt",
            "请解释行为方式",
        )

        self.assertIsNone(
            run.draft
        )

        self.assertEqual(
            len(run.verified),
            0,
        )

        self.assertEqual(
            run.rejected[0]["status"],
            "text_not_located",
        )

        self.assertEqual(
            len(model.calls),
            0,
        )

    # ========================================
    # 测试4：不存在的来源不能检索
    # ========================================

    def test_wrong_source_fails_before_retrieval(
        self,
    ):

        search = FakeSearch()

        with self.assertRaisesRegex(
            ValueError,
            "来源",
        ):

            GroundedMethodCoach(
                search,
                self.catalog,
                FakeModel(output()),
                FakeMethods(),
            ).run(
                "yagi_self_understanding",
                "unknown.txt",
                "探索兴趣",
            )

        self.assertEqual(
            search.calls,
            [],
        )

    # ========================================
    # 测试5：不能引用虚构证据编号
    # ========================================

    def test_wrong_evidence_id_rejected(self):

        with self.assertRaisesRegex(
            ValueError,
            "证据编号",
        ):

            GroundedMethodCoach(
                FakeSearch(),
                self.catalog,
                FakeModel(
                    output(
                        ids=["E999"]
                    )
                ),
                FakeMethods(),
                attempts=1,
            ).run(
                "yagi_self_understanding",
                "course.txt",
                "探索兴趣",
            )

    # ========================================
    # 测试6：允许明确报告证据不足
    # ========================================

    def test_insufficient_evidence_can_be_declared(
        self,
    ):

        run = GroundedMethodCoach(
            FakeSearch(),
            self.catalog,
            FakeModel(
                output(
                    "insufficient_evidence"
                )
            ),
            FakeMethods(),
        ).run(
            "contradiction_lens",
            "course.txt",
            "当前现实问题怎样分析？",
        )

        self.assertEqual(
            run.draft.status,
            "insufficient_evidence",
        )

        self.assertEqual(
            run.draft.evidence_ids,
            [],
        )


if __name__ == "__main__":
    unittest.main()
