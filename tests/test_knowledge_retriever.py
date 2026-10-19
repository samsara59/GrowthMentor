
import tempfile
import unittest

from pathlib import Path

from app.knowledge.schema import Evidence, KnowledgeItem
from app.knowledge.review_pipeline import ReviewResult
from app.knowledge.store import save_review_results
from app.rag.retriever import search_knowledge


class TestKnowledgeRetriever(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.db_path = (
            Path(self.temp_dir.name) / "knowledge.db"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_review(
        self,
        concept,
        statement,
        status="supported",
    ):
        item = KnowledgeItem(
            concept=concept,
            statement=statement,
            knowledge_type="definition",
            evidence=[
                Evidence(
                    source="demo.pdf",
                    location="第1页",
                    quote=statement,
                )
            ],
            extraction_method="local_llm",
        )

        return ReviewResult(
            item=item,
            status=status,
            reason="测试审核结果",
        )

    def test_search_by_concept(self):
        save_review_results(
            self.db_path,
            [
                self.make_review(
                    "学习力",
                    "学习力涉及知识获取。",
                )
            ],
        )

        hits = search_knowledge(
            self.db_path,
            "学习力",
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].score, 3)

    def test_search_by_statement(self):
        save_review_results(
            self.db_path,
            [
                self.make_review(
                    "学习力",
                    "个体能够理解和运用知识。",
                )
            ],
        )

        hits = search_knowledge(
            self.db_path,
            "运用知识",
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].score, 1)

    def test_exclude_unsupported(self):
        save_review_results(
            self.db_path,
            [
                self.make_review(
                    "学习力",
                    "一条不被原文支持的陈述",
                    status="unsupported",
                )
            ],
        )

        hits = search_knowledge(
            self.db_path,
            "学习力",
        )

        self.assertEqual(hits, [])

    def test_ranking(self):
        save_review_results(
            self.db_path,
            [
                self.make_review(
                    "学习力训练",
                    "关于训练方法的知识。",
                ),
                self.make_review(
                    "学习力",
                    "关于学习力的定义。",
                ),
            ],
        )

        hits = search_knowledge(
            self.db_path,
            "学习力",
        )

        self.assertEqual(len(hits), 2)

        self.assertEqual(
            hits[0].result.item.concept,
            "学习力",
        )

    def test_preserve_source(self):
        save_review_results(
            self.db_path,
            [
                self.make_review(
                    "Question",
                    "Question是指问题。",
                )
            ],
        )

        hits = search_knowledge(
            self.db_path,
            "Question",
        )

        self.assertEqual(
            hits[0].result.item.evidence[0].source,
            "demo.pdf",
        )

        self.assertEqual(
            hits[0].result.item.evidence[0].location,
            "第1页",
        )

    def test_empty_query(self):
        save_review_results(self.db_path, [])

        with self.assertRaises(ValueError):
            search_knowledge(self.db_path, "   ")


if __name__ == "__main__":
    unittest.main()
