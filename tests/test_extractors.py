
import unittest

from app.ingestion.schema import DocumentRecord
from app.knowledge.schema import Evidence, KnowledgeItem
from app.knowledge.extractors import (
    RuleExtractor,
    run_extractor,
)


class FakeLLMExtractor:
    """模拟一个产生错误引用的大模型。"""

    def extract(self, record):
        return [
            KnowledgeItem(
                concept="学习力",
                statement="一条待核验的知识",
                knowledge_type="definition",
                evidence=[
                    Evidence(
                        source=record.source,
                        location=record.location,
                        quote="教材中不存在的定义。"
                    )
                ],
                extraction_method="local_llm"
            )
        ]


class TestExtractors(unittest.TestCase):

    def setUp(self):
        self.record = DocumentRecord(
            source="demo.pdf",
            location="第3页",
            text="学习力是指个体获取和运用知识的能力。",
            method="pdf_text"
        )

    def test_rule_extractor(self):
        extractor = RuleExtractor()

        items = extractor.extract(self.record)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].concept, "学习力")

    def test_accept_valid_knowledge(self):
        accepted, rejected = run_extractor(
            RuleExtractor(),
            self.record
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 0)

    def test_reject_fake_evidence(self):
        accepted, rejected = run_extractor(
            FakeLLMExtractor(),
            self.record
        )

        self.assertEqual(len(accepted), 0)
        self.assertEqual(len(rejected), 1)


if __name__ == "__main__":
    unittest.main()
