
import unittest

from app.ingestion.schema import DocumentRecord
from app.knowledge.rule_extractor import extract_definitions


class TestRuleExtractor(unittest.TestCase):

    def test_extract_definition(self):
        record = DocumentRecord(
            source="demo.pdf",
            location="第3页",
            text="学习力是指个体获取和运用知识的能力。",
            method="pdf_text"
        )

        items = extract_definitions(record)

        self.assertEqual(len(items), 1)

        item = items[0]

        self.assertEqual(item.concept, "学习力")
        self.assertEqual(item.knowledge_type, "definition")
        self.assertEqual(item.extraction_method, "rule")

        self.assertEqual(
            item.evidence[0].source,
            "demo.pdf"
        )

        self.assertEqual(
            item.evidence[0].location,
            "第3页"
        )

        self.assertIn(
            item.evidence[0].quote,
            record.text
        )

    def test_no_explicit_definition(self):
        record = DocumentRecord(
            source="demo.pdf",
            location="第4页",
            text="学习力对个人成长非常重要。",
            method="pdf_text"
        )

        items = extract_definitions(record)

        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
