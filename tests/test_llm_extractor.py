
import json
import unittest

from app.ingestion.schema import DocumentRecord
from app.knowledge.llm_extractor import LLMExtractor
from app.knowledge.extractors import run_extractor


class FakeClient:
    """模拟模型客户端，不需要联网。"""

    def __init__(self, response):
        self.response = response
        self.last_prompt = None

    def generate(self, prompt):
        self.last_prompt = prompt
        return self.response


class TestLLMExtractor(unittest.TestCase):

    def setUp(self):
        self.quote = "学习力是指个体获取和运用知识的能力。"

        self.record = DocumentRecord(
            source="demo.pdf",
            location="第3页",
            text=self.quote,
            method="pdf_text",
        )

        self.valid_response = json.dumps(
            [
                {
                    "concept": "学习力",
                    "statement": self.quote,
                    "knowledge_type": "definition",
                    "quotes": [self.quote],
                }
            ],
            ensure_ascii=False,
        )

    def test_valid_output(self):
        client = FakeClient(self.valid_response)

        extractor = LLMExtractor(
            client=client,
            method="local_llm",
        )

        items = extractor.extract(self.record)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].concept, "学习力")
        self.assertEqual(
            items[0].extraction_method,
            "local_llm",
        )
        self.assertEqual(
            items[0].evidence[0].source,
            "demo.pdf",
        )
        self.assertIn(self.quote, client.last_prompt)

    def test_cloud_method(self):
        client = FakeClient(self.valid_response)

        extractor = LLMExtractor(
            client=client,
            method="cloud_llm",
        )

        items = extractor.extract(self.record)

        self.assertEqual(
            items[0].extraction_method,
            "cloud_llm",
        )

    def test_reject_fake_quote(self):
        response = json.dumps(
            [
                {
                    "concept": "学习力",
                    "statement": "一条待验证的陈述",
                    "knowledge_type": "definition",
                    "quotes": ["教材中不存在的原文。"],
                }
            ],
            ensure_ascii=False,
        )

        extractor = LLMExtractor(
            FakeClient(response),
            method="local_llm",
        )

        accepted, rejected = run_extractor(
            extractor,
            self.record,
        )

        self.assertEqual(len(accepted), 0)
        self.assertEqual(len(rejected), 1)

    def test_invalid_json(self):
        extractor = LLMExtractor(
            FakeClient("not valid json"),
            method="local_llm",
        )

        with self.assertRaises(ValueError):
            extractor.extract(self.record)

    def test_empty_response(self):
        extractor = LLMExtractor(
            FakeClient("[]"),
            method="local_llm",
        )

        self.assertEqual(
            extractor.extract(self.record),
            [],
        )

    def test_single_object_output(self):
        # 模拟 Gemma 3 返回单个 JSON 对象
        response = json.dumps(
            {
                "concept": "学习力",
                "statement": self.quote,
                "knowledge_type": "definition",
                "quotes": [self.quote],
            },
            ensure_ascii=False,
        )

        extractor = LLMExtractor(
            client=FakeClient(response),
            method="local_llm",
        )

        accepted, rejected = run_extractor(
            extractor,
            self.record,
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 0)
        self.assertEqual(
            accepted[0].concept,
            "学习力",
        )


    def test_wrapped_knowledge_objects(self):

        second_quote = (
            "复盘是指对过去行动进行回顾并总结经验的过程。"
        )

        record = DocumentRecord(
            source="demo.pdf",
            location="第1页",
            text=self.quote + "\n" + second_quote,
            method="pdf_text",
        )

        response = json.dumps(
            {
                "knowledge_objects": [
                    {
                        "concept": "学习力",
                        "statement": self.quote,
                        "knowledge_type": "definition",
                        "quotes": [self.quote],
                    },
                    {
                        "concept": "复盘",
                        "statement": second_quote,
                        "knowledge_type": "definition",
                        "quotes": [second_quote],
                    },
                ]
            },
            ensure_ascii=False,
        )

        extractor = LLMExtractor(
            client=FakeClient(response),
            method="local_llm",
        )

        accepted, rejected = run_extractor(
            extractor,
            record,
        )

        self.assertEqual(len(accepted), 2)
        self.assertEqual(len(rejected), 0)

        concepts = {item.concept for item in accepted}

        self.assertEqual(
            concepts,
            {"学习力", "复盘"},
        )


if __name__ == "__main__":
    unittest.main()
