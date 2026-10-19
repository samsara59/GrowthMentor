
import unittest

from app.ingestion.schema import DocumentRecord
from app.knowledge.schema import Evidence, KnowledgeItem

from app.knowledge.semantic_validator import SemanticJudgment
from app.knowledge.review_pipeline import review_record


class FakeExtractor:
    def __init__(self, items):
        self.items = items

    def extract(self, record):
        return self.items


class FakeValidator:
    def __init__(self, verdict="supported", fail=False):
        self.verdict = verdict
        self.fail = fail
        self.calls = 0

    def validate(self, item, records):
        self.calls += 1

        if self.fail:
            raise RuntimeError("模拟模型调用失败")

        return SemanticJudgment(
            verdict=self.verdict,
            reason="模拟语义判断",
        )


class TestReviewPipeline(unittest.TestCase):

    def setUp(self):
        self.quote = "学习力是指获取和运用知识的能力。"

        self.record = DocumentRecord(
            source="demo.pdf",
            location="第1页",
            text=self.quote,
            method="pdf_text",
        )

    def make_item(self, quote=None):
        return KnowledgeItem(
            concept="学习力",
            statement="学习力涉及知识获取和运用。",
            knowledge_type="definition",
            evidence=[
                Evidence(
                    source="demo.pdf",
                    location="第1页",
                    quote=quote if quote is not None else self.quote,
                )
            ],
            extraction_method="local_llm",
        )

    def test_supported(self):
        validator = FakeValidator("supported")

        results = review_record(
            self.record,
            FakeExtractor([self.make_item()]),
            validator,
        )

        self.assertEqual(results[0].status, "supported")
        self.assertEqual(validator.calls, 1)

    def test_unsupported(self):
        results = review_record(
            self.record,
            FakeExtractor([self.make_item()]),
            FakeValidator("unsupported"),
        )

        self.assertEqual(results[0].status, "unsupported")

    def test_uncertain(self):
        results = review_record(
            self.record,
            FakeExtractor([self.make_item()]),
            FakeValidator("uncertain"),
        )

        self.assertEqual(results[0].status, "uncertain")

    def test_invalid_evidence(self):
        validator = FakeValidator()

        results = review_record(
            self.record,
            FakeExtractor([
                self.make_item("不存在的引用")
            ]),
            validator,
        )

        self.assertEqual(
            results[0].status,
            "evidence_invalid",
        )

        # 引用不存在，不应该浪费模型调用
        self.assertEqual(validator.calls, 0)

    def test_model_failure(self):
        results = review_record(
            self.record,
            FakeExtractor([self.make_item()]),
            FakeValidator(fail=True),
        )

        self.assertEqual(
            results[0].status,
            "review_error",
        )


if __name__ == "__main__":
    unittest.main()
