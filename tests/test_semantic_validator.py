
import json
import unittest

from app.ingestion.schema import DocumentRecord
from app.knowledge.schema import Evidence, KnowledgeItem

from app.knowledge.semantic_validator import (
    SemanticValidator,
)


class FakeClient:
    """模拟语义核验模型响应。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.last_prompt = None

    def generate(self, prompt):
        self.calls += 1
        self.last_prompt = prompt
        return self.response


class TestSemanticValidator(unittest.TestCase):

    def setUp(self):
        self.quote = (
            "学习力是指个体获取和运用知识的能力。"
        )

        self.record = DocumentRecord(
            source="demo.pdf",
            location="第1页",
            text=self.quote,
            method="pdf_text",
        )

    def make_item(self, statement, quote=None):
        return KnowledgeItem(
            concept="学习力",
            statement=statement,
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

    def make_client(self, verdict, reason):
        response = json.dumps({
            "verdict": verdict,
            "reason": reason,
        }, ensure_ascii=False)

        return FakeClient(response)

    def test_supported(self):
        client = self.make_client(
            "supported",
            "引文直接支持陈述",
        )

        validator = SemanticValidator(client)

        item = self.make_item(self.quote)

        result = validator.validate(
            item, [self.record]
        )

        self.assertEqual(result.verdict, "supported")
        self.assertEqual(client.calls, 1)
        self.assertIn(self.quote, client.last_prompt)

    def test_unsupported(self):
        client = self.make_client(
            "unsupported",
            "引文没有提供未来收入预测的依据",
        )

        validator = SemanticValidator(client)

        item = self.make_item(
            "学习力能够准确预测未来收入。"
        )

        result = validator.validate(
            item, [self.record]
        )

        self.assertEqual(result.verdict, "unsupported")

    def test_uncertain(self):
        client = self.make_client(
            "uncertain",
            "缺少足够信息",
        )

        validator = SemanticValidator(client)

        item = self.make_item(
            "学习力包含多个具体阶段。"
        )

        result = validator.validate(
            item, [self.record]
        )

        self.assertEqual(result.verdict, "uncertain")

    def test_invalid_evidence(self):
        client = self.make_client(
            "supported",
            "测试响应",
        )

        validator = SemanticValidator(client)

        item = self.make_item(
            "错误引用",
            quote="教材不存在的引用",
        )

        with self.assertRaises(ValueError):
            validator.validate(item, [self.record])

        # 引用不存在时，不能调用语义模型
        self.assertEqual(client.calls, 0)

    def test_invalid_model_response(self):
        client = FakeClient(
            '{"verdict":"maybe","reason":"无法判断"}'
        )

        validator = SemanticValidator(client)
        item = self.make_item(self.quote)

        with self.assertRaises(ValueError):
            validator.validate(item, [self.record])


if __name__ == "__main__":
    unittest.main()
