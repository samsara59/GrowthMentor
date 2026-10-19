
import json
import unittest

from app.knowledge.schema import (
    Evidence,
    KnowledgeItem,
)
from app.knowledge.review_pipeline import ReviewResult
from app.rag.vector_store import SemanticHit
from app.rag.pipeline import answer_question


class FakeRetriever:
    """模拟检索器。"""

    def __init__(self, hits):
        self.hits = hits
        self.calls = 0

    def search(self, query, limit=5):
        self.calls += 1
        return self.hits[:limit]


class FakeGenerator:
    """模拟 Gemma 3。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.last_prompt = None

    def generate(self, prompt):
        self.calls += 1
        self.last_prompt = prompt

        return json.dumps(
            self.response,
            ensure_ascii=False,
        )


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.quote = "Question可以是学习者想到的问题。"

        item = KnowledgeItem(
            concept="Question",
            statement=self.quote,
            knowledge_type="definition",
            evidence=[
                Evidence(
                    source="demo.pdf",
                    location="第3页",
                    quote=self.quote,
                )
            ],
            extraction_method="local_llm",
        )

        review = ReviewResult(
            item=item,
            status="supported",
            reason="测试审核结果",
        )

        self.hit = SemanticHit(
            result=review,
            distance=0.15,
        )

    def test_answer_with_citation(self):
        retriever = FakeRetriever([self.hit])

        generator = FakeGenerator({
            "answer": "Question是学习者想到的问题。",
            "status": "answered",
            "used_evidence_ids": ["K1"],
        })

        result = answer_question(
            "Question是什么意思？",
            retriever,
            generator,
        )

        self.assertEqual(result.status, "answered")
        self.assertEqual(len(result.citations), 1)

        self.assertEqual(
            result.citations[0].source,
            "demo.pdf",
        )

        self.assertEqual(
            result.citations[0].location,
            "第3页",
        )

        self.assertIn(self.quote, generator.last_prompt)

    def test_no_evidence(self):
        retriever = FakeRetriever([])

        generator = FakeGenerator({
            "answer": "随便生成的回答",
            "status": "answered",
            "used_evidence_ids": [],
        })

        result = answer_question(
            "没有相关知识的问题",
            retriever,
            generator,
        )

        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.citations, [])

        # 没有证据时，不调用模型
        self.assertEqual(generator.calls, 0)

    def test_reject_fake_citation(self):
        retriever = FakeRetriever([self.hit])

        generator = FakeGenerator({
            "answer": "错误引用",
            "status": "answered",
            "used_evidence_ids": ["K99"],
        })

        with self.assertRaises(ValueError):
            answer_question(
                "Question是什么？",
                retriever,
                generator,
            )

    def test_partial_answer(self):
        retriever = FakeRetriever([self.hit])

        generator = FakeGenerator({
            "answer": (
                "当前只能说明Question的含义，"
                "不足以说明全部QAS步骤。"
            ),
            "status": "partial",
            "used_evidence_ids": ["K1"],
        })

        result = answer_question(
            "QAS的全部步骤是什么？",
            retriever,
            generator,
        )

        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.citations), 1)

    def test_distance_filter(self):
        far_hit = SemanticHit(
            result=self.hit.result,
            distance=0.95,
        )

        retriever = FakeRetriever([far_hit])

        generator = FakeGenerator({
            "answer": "不应该被调用",
            "status": "answered",
            "used_evidence_ids": ["K1"],
        })

        result = answer_question(
            "无关的问题",
            retriever,
            generator,
            max_distance=0.6,
        )

        self.assertEqual(result.status, "insufficient")
        self.assertEqual(generator.calls, 0)

    def test_missing_citation_id(self):
        retriever = FakeRetriever([self.hit])

        generator = FakeGenerator({
            "answer": "没有引用的回答",
            "status": "answered",
            "used_evidence_ids": [],
        })

        with self.assertRaises(ValueError):
            answer_question(
                "Question是什么？",
                retriever,
                generator,
            )


if __name__ == "__main__":
    unittest.main()
