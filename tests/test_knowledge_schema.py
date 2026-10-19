
import unittest

from pydantic import ValidationError

from app.knowledge.schema import Evidence, KnowledgeItem


class TestKnowledgeSchema(unittest.TestCase):

    def test_valid_knowledge(self):
        item = KnowledgeItem(
            concept="示例概念",
            statement="一条测试知识",
            knowledge_type="definition",
            evidence=[
                Evidence(
                    source="demo.pdf",
                    location="第3页",
                    quote="测试原文"
                )
            ],
            extraction_method="rule"
        )

        self.assertEqual(item.concept, "示例概念")
        self.assertEqual(item.evidence[0].source, "demo.pdf")

    def test_missing_evidence(self):
        with self.assertRaises(ValidationError):
            KnowledgeItem(
                concept="示例概念",
                statement="缺少证据的知识",
                knowledge_type="definition",
                evidence=[],
                extraction_method="local_llm"
            )


if __name__ == "__main__":
    unittest.main()
