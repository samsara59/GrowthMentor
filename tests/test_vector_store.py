
import tempfile
import unittest

from pathlib import Path

from app.knowledge.schema import Evidence, KnowledgeItem
from app.knowledge.review_pipeline import ReviewResult
from app.knowledge.store import save_review_results
from app.rag.vector_store import ChromaKnowledgeStore


class FakeEmbedder:
    """测试专用 Embedding，不调用真实模型。"""

    model = "fake-embedding-v1"

    def embed(self, texts):
        vectors = []

        for text in texts:
            lower_text = text.casefold()

            if (
                "question" in lower_text
                or "问题" in text
                or "疑问" in text
            ):
                vectors.append([1.0, 0.0, 0.0])

            elif (
                "source" in lower_text
                or "来源" in text
            ):
                vectors.append([0.0, 1.0, 0.0])

            else:
                vectors.append([0.0, 0.0, 1.0])

        return vectors


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True
        )

        root = Path(self.temp.name)

        self.db_path = root / "knowledge.db"
        self.vector_path = root / "chroma"

        self.store = ChromaKnowledgeStore(
            db_path=self.db_path,
            vector_path=self.vector_path,
            embedder=FakeEmbedder(),
        )

    def tearDown(self):
        self.temp.cleanup()

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
            extraction_method="rule",
        )

        return ReviewResult(
            item=item,
            status=status,
            reason="测试审核结果",
        )

    def test_sync_and_search(self):
        reviews = [
            self.make_review(
                "Question",
                "Question 是学习中提出的问题。",
            ),
            self.make_review(
                "Source",
                "Source 表示信息来源。",
            ),
        ]

        save_review_results(self.db_path, reviews)

        count = self.store.sync()

        self.assertEqual(count, 2)
        self.assertEqual(self.store.collection.count(), 2)

        hits = self.store.search(
            "阅读时如何发现问题？"
        )

        self.assertEqual(len(hits), 2)
        self.assertEqual(
            hits[0].result.item.concept,
            "Question",
        )

    def test_exclude_unsupported(self):
        review = self.make_review(
            "Question",
            "Question 是学习中的问题。",
            status="unsupported",
        )

        save_review_results(self.db_path, [review])

        count = self.store.sync()

        self.assertEqual(count, 0)
        self.assertEqual(self.store.collection.count(), 0)

    def test_revoked_knowledge(self):
        original = self.make_review(
            "Question",
            "Question 是学习中的问题。",
        )

        save_review_results(self.db_path, [original])
        self.store.sync()

        self.assertEqual(self.store.collection.count(), 1)

        # 同一知识的审核状态被撤销
        revoked = self.make_review(
            "Question",
            "Question 是学习中的问题。",
            status="unsupported",
        )

        save_review_results(self.db_path, [revoked])

        # Chroma 还没同步，但检索必须立即排除
        hits = self.store.search("Question")

        self.assertEqual(hits, [])

        # 同步后旧向量也应被删除
        self.store.sync()

        self.assertEqual(self.store.collection.count(), 0)

    def test_persistence(self):
        review = self.make_review(
            "Question",
            "Question 是学习中的问题。",
        )

        save_review_results(self.db_path, [review])

        self.store.sync()

        # 使用相同磁盘路径重新打开 Chroma
        reopened = ChromaKnowledgeStore(
            db_path=self.db_path,
            vector_path=self.vector_path,
            embedder=FakeEmbedder(),
        )

        hits = reopened.search("Question")

        self.assertEqual(len(hits), 1)
        self.assertEqual(
            hits[0].result.item.evidence[0].source,
            "demo.pdf",
        )


if __name__ == "__main__":
    unittest.main()
