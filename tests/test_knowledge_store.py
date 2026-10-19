
import tempfile
import unittest

from pathlib import Path

from app.knowledge.schema import Evidence, KnowledgeItem
from app.knowledge.review_pipeline import ReviewResult

from app.knowledge.store import (
    save_review_results,
    load_review_results,
)


class TestKnowledgeStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.db_path = (
            Path(self.temp_dir.name) / "knowledge.db"
        )

        self.item = KnowledgeItem(
            concept="学习力",
            statement="学习力涉及知识获取和运用。",
            knowledge_type="definition",
            evidence=[
                Evidence(
                    source="demo.pdf",
                    location="第1页",
                    quote="学习力是指获取和运用知识的能力。",
                )
            ],
            extraction_method="local_llm",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_result(self, status="supported"):
        return ReviewResult(
            item=self.item,
            status=status,
            reason="测试审核理由",
        )

    def test_save_and_load(self):
        save_review_results(
            self.db_path,
            [self.make_result()],
        )

        loaded = load_review_results(self.db_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded[0].item.concept,
            "学习力",
        )
        self.assertEqual(
            loaded[0].status,
            "supported",
        )

    def test_preserve_evidence(self):
        save_review_results(
            self.db_path,
            [self.make_result()],
        )

        loaded = load_review_results(self.db_path)

        self.assertEqual(
            loaded[0].item.evidence[0].source,
            "demo.pdf",
        )
        self.assertEqual(
            loaded[0].item.evidence[0].location,
            "第1页",
        )

    def test_no_exact_duplicates(self):
        result = self.make_result()

        save_review_results(self.db_path, [result])
        save_review_results(self.db_path, [result])

        loaded = load_review_results(self.db_path)

        self.assertEqual(len(loaded), 1)

    def test_update_review_status(self):
        save_review_results(
            self.db_path,
            [self.make_result("uncertain")],
        )

        save_review_results(
            self.db_path,
            [self.make_result("supported")],
        )

        loaded = load_review_results(self.db_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded[0].status,
            "supported",
        )

    def test_filter_by_status(self):
        save_review_results(
            self.db_path,
            [self.make_result("unsupported")],
        )

        loaded = load_review_results(
            self.db_path,
            status="supported",
        )

        self.assertEqual(loaded, [])


if __name__ == "__main__":
    unittest.main()
