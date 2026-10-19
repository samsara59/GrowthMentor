
import json
import unittest

from app.knowledge.schema import Evidence, KnowledgeItem
from app.knowledge.review_pipeline import ReviewResult
from app.rag.vector_store import SemanticHit

from app.rag.answerability import (
    CoverageAssessment,
    guarded_answer_question,
)


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, limit=5):
        return self.hits[:limit]


class FakeJudge:
    def __init__(self, assessment):
        self.assessment = assessment
        self.calls = 0

    def assess(self, question, hits):
        self.calls += 1
        return self.assessment


class FakeGenerator:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return json.dumps(
            self.response,
            ensure_ascii=False,
        )


class TestAnswerability(unittest.TestCase):

    def setUp(self):
        item = KnowledgeItem(
            concept="Question",
            statement="Question是指学习时想到的问题。",
            knowledge_type="definition",
            evidence=[
                Evidence(
                    source="demo.pdf",
                    location="第1页",
                    quote="Question是指学习时想到的问题。",
                )
            ],
            extraction_method="local_llm",
        )

        review = ReviewResult(
            item=item,
            status="supported",
            reason="测试",
        )

        self.hit = SemanticHit(
            result=review,
            distance=0.2,
        )

    def make_generator(self):
        return FakeGenerator({
            "answer": "Question是学习时想到的问题。",
            "status": "answered",
            "used_evidence_ids": ["K1"],
        })

    def test_complete(self):
        judge = FakeJudge(
            CoverageAssessment(
                status="complete",
                relevant_evidence_ids=["K1"],
                supported_parts=["Question的定义"],
                missing_parts=[],
                reason="证据充分",
            )
        )

        result = guarded_answer_question(
            "Question是什么？",
            FakeRetriever([self.hit]),
            self.make_generator(),
            judge,
        )

        self.assertEqual(result.status, "answered")
        self.assertEqual(len(result.citations), 1)

    def test_partial_overrides_answered(self):
        judge = FakeJudge(
            CoverageAssessment(
                status="partial",
                relevant_evidence_ids=["K1"],
                supported_parts=["Question的定义"],
                missing_parts=["Answer的记录方法", "Source的记录方法"],
                reason="只能回答部分问题",
            )
        )

        result = guarded_answer_question(
            "QAS的完整使用方法是什么？",
            FakeRetriever([self.hit]),
            self.make_generator(),
            judge,
        )

        self.assertEqual(result.status, "partial")
        self.assertIn("Answer的记录方法", result.answer)

    def test_none_skips_generator(self):
        judge = FakeJudge(
            CoverageAssessment(
                status="none",
                relevant_evidence_ids=[],
                supported_parts=[],
                missing_parts=["天气预报信息"],
                reason="教材证据与天气无关",
            )
        )

        generator = self.make_generator()

        result = guarded_answer_question(
            "明天天气怎么样？",
            FakeRetriever([self.hit]),
            generator,
            judge,
        )

        self.assertEqual(result.status, "insufficient")
        self.assertEqual(generator.calls, 0)
        self.assertEqual(result.citations, [])

    def test_fake_evidence_id(self):
        judge = FakeJudge(
            CoverageAssessment(
                status="complete",
                relevant_evidence_ids=["K99"],
                supported_parts=["Question"],
                missing_parts=[],
                reason="错误编号",
            )
        )

        with self.assertRaises(ValueError):
            guarded_answer_question(
                "Question是什么？",
                FakeRetriever([self.hit]),
                self.make_generator(),
                judge,
            )

    def test_distance_filter(self):
        judge = FakeJudge(
            CoverageAssessment(
                status="complete",
                relevant_evidence_ids=["K1"],
                supported_parts=["Question"],
                missing_parts=[],
                reason="证据充分",
            )
        )

        generator = self.make_generator()

        result = guarded_answer_question(
            "测试问题",
            FakeRetriever([self.hit]),
            generator,
            judge,
            max_distance=0.1,
        )

        self.assertEqual(result.status, "insufficient")
        self.assertEqual(judge.calls, 0)
        self.assertEqual(generator.calls, 0)


if __name__ == "__main__":
    unittest.main()
