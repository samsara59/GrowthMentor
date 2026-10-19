
"""第70课：教材检索、来源过滤和证据编号测试。"""

import json
import unittest

from app.agent.evidence_coach import (
    Evidence,
    EvidenceCoach,
    SourceEvidenceSearch,
    list_indexed_sources,
)


# ============================================
# 1. 模拟Chroma
# ============================================

class FakeCollection:

    def count(self):
        return 3

    def get(self, **kwargs):

        if kwargs.get("where"):

            name = kwargs[
                "where"
            ]["source"]

            return {
                "ids": (
                    ["1"]
                    if name == "A.docx"
                    else []
                ),

                "metadatas": (
                    [{"source": name}]
                    if name == "A.docx"
                    else []
                ),
            }

        return {
            "ids": [
                "1", "2", "3"
            ],

            "metadatas": [
                {"source": "A.docx"},
                {"source": "B.pdf"},
                {},
            ],
        }

    def query(self, **kwargs):

        return {
            "documents": [
                ["原文测试片段"]
            ],

            "metadatas": [
                [{
                    "source": "A.docx",
                    "location": "page=2",
                }]
            ],

            "distances": [
                [0.35]
            ],
        }


# ============================================
# 2. 模拟Gemma 3
# ============================================

class FakeModel:

    def __init__(self, data):

        self.data = data

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

        return self.data


# ============================================
# 3. 合法的测试回复
# ============================================

def good():

    return {
        "interpretation": (
            "候选片段可能讨论某种思考方法，"
            "但仍需核对原文。"
        ),

        "application": (
            "请回忆一件已经解决的问题，"
            "记录当时最自然的思考动作。"
        ),

        "follow_up": (
            "你最近主动研究过"
            "哪个具体问题？"
        ),

        "limitations": (
            "来源片段未逐字核验，"
            "不足以归因完整理论"
            "或认定兴趣。"
        ),

        "evidence_ids": [
            "E1"
        ],
    }


# ============================================
# 4. 开始测试
# ============================================

class TestEvidenceCoach(unittest.TestCase):

    def setUp(self):

        self.evidence = [
            Evidence(
                id="E1",

                source="A.docx",

                location="page=2",

                excerpt="原文测试片段",

                distance=0.35,
            )
        ]

    # ========================================
    # 测试1：列出真实来源
    # ========================================

    def test_source_listing(self):

        names, all_scanned = (
            list_indexed_sources(
                FakeCollection()
            )
        )

        self.assertEqual(
            names,
            ["A.docx", "B.pdf"],
        )

        self.assertTrue(
            all_scanned
        )

    # ========================================
    # 测试2：精确限定教材
    # ========================================

    def test_scope_exact_source(self):

        search = SourceEvidenceSearch(
            FakeCollection(),
            lambda query: [
                0.1, 0.2
            ],
        )

        result = search.search(
            "怎么学习",
            source="A.docx",
        )

        self.assertEqual(
            result[0].source,
            "A.docx",
        )

        self.assertEqual(
            result[0].id,
            "E1",
        )

        self.assertEqual(
            search.search(
                "怎么学习",
                source="missing",
            ),
            [],
        )

    # ========================================
    # 测试3：真正把证据传给LLM
    # ========================================

    def test_grounded_answer(self):

        model = FakeModel(
            good()
        )

        answer = EvidenceCoach(
            model
        ).answer(
            "我喜欢什么",
            self.evidence,
        )

        self.assertEqual(
            answer.evidence_ids,
            ["E1"],
        )

        self.assertIn(
            "retrieved_candidates_unverified",
            model.calls[0],
        )

    # ========================================
    # 测试4：禁止虚假证据编号
    # ========================================

    def test_reject_invented_citation(self):

        output = good()

        output["evidence_ids"] = [
            "E999"
        ]

        with self.assertRaisesRegex(
            ValueError,
            "不存在",
        ):

            EvidenceCoach(
                FakeModel(output),
                attempts=1,
            ).answer(
                "问题",
                self.evidence,
            )

    # ========================================
    # 测试5：没有证据不得冒充教材指导
    # ========================================

    def test_no_evidence_no_attribution(self):

        with self.assertRaisesRegex(
            ValueError,
            "没有带来源",
        ):

            EvidenceCoach(
                FakeModel(good())
            ).answer(
                "问题",
                [],
            )


if __name__ == "__main__":
    unittest.main()
