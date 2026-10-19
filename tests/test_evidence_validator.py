
import unittest

from app.ingestion.schema import DocumentRecord
from app.knowledge.schema import Evidence, KnowledgeItem
from app.knowledge.rule_extractor import extract_definitions
from app.knowledge.evidence_validator import (
    validate_evidence,
    validate_knowledge,
)


class TestEvidenceValidator(unittest.TestCase):

    def setUp(self):
        self.record = DocumentRecord(
            source="demo.pdf",
            location="第3页",
            text="学习力是指个体获取和运用知识的能力。",
            method="pdf_text"
        )

    def test_valid_evidence(self):
        evidence = Evidence(
            source="demo.pdf",
            location="第3页",
            quote="学习力是指个体获取和运用知识的能力。"
        )

        self.assertTrue(
            validate_evidence(evidence, [self.record])
        )

    def test_fake_quote(self):
        evidence = Evidence(
            source="demo.pdf",
            location="第3页",
            quote="学习力可以准确预测未来。"
        )

        self.assertFalse(
            validate_evidence(evidence, [self.record])
        )

    def test_wrong_location(self):
        evidence = Evidence(
            source="demo.pdf",
            location="第5页",
            quote="学习力是指个体获取和运用知识的能力。"
        )

        self.assertFalse(
            validate_evidence(evidence, [self.record])
        )

    def test_rule_extractor_integration(self):
        items = extract_definitions(self.record)

        self.assertEqual(len(items), 1)

        self.assertTrue(
            validate_knowledge(items[0], [self.record])
        )


if __name__ == "__main__":
    unittest.main()
