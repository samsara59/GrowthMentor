
import json
import sqlite3
import tempfile
import unittest

from pathlib import Path

from app.knowledge.coverage_audit import (
    audit_knowledge,
    filename_from_record,
    scan_sqlite,
)


# ============================================
# 1. 模拟一个Chroma Collection
# ============================================

class FakeCollection:

    def __init__(self, metadatas):
        self.metadatas = metadatas

    def count(self):
        return len(self.metadatas)

    def get(
        self,
        *,
        limit,
        offset,
        include,
    ):
        return {
            "metadatas": self.metadatas[
                offset:offset + limit
            ]
        }


# ============================================
# 2. 模拟Chroma客户端
# ============================================

class FakeChroma:

    def list_collections(self):
        return ["main"]

    def get_collection(self, *, name):

        return FakeCollection([
            {
                "source": r"D:\books\分析力.pdf"
            },
            {
                "source": "行动力.docx"
            },
        ])


# ============================================
# 3. 测试知识库审计
# ============================================

class TestKnowledgeAudit(unittest.TestCase):

    # ----------------------------------------
    # 测试1：Windows路径识别
    # ----------------------------------------

    def test_source_from_windows_path(self):

        result = filename_from_record({
            "metadata": {
                "source": r"D:\books\分析力.pdf"
            }
        })

        self.assertEqual(
            result,
            "分析力.pdf",
        )

    # ----------------------------------------
    # 测试2：不能编造来源
    # ----------------------------------------

    def test_no_invented_source(self):

        result = filename_from_record({
            "text": "我会分析问题"
        })

        self.assertIsNone(result)

    # ----------------------------------------
    # 测试3：只读访问SQLite
    # ----------------------------------------

    def test_sqlite_read_only(self):

        with tempfile.TemporaryDirectory() as temp:

            db = Path(temp) / "knowledge.db"

            conn = sqlite3.connect(db)

            try:
                with conn:

                    conn.execute(
                        "CREATE TABLE knowledge_items "
                        "(id INTEGER PRIMARY KEY, "
                        "content TEXT)"
                    )

                    conn.execute(
                        "INSERT INTO knowledge_items"
                        "(content) VALUES (?)",
                        ("测试文本",),
                    )

            finally:
                conn.close()

            result = scan_sqlite(db)

            self.assertEqual(
                result["status"],
                "ok",
            )

            self.assertEqual(
                result["tables"][0]["rows"],
                1,
            )

    # ----------------------------------------
    # 测试4：各阶段覆盖比较
    # ----------------------------------------

    def test_coverage_comparison(self):

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            books = root / "books"
            books.mkdir()

            (root / "data" / "chroma").mkdir(
                parents=True
            )

            # 三份原始教材
            for filename in (
                "分析力.pdf",
                "行动力.docx",
                "价值观.pdf",
            ):

                (books / filename).write_bytes(
                    b"test"
                )

            # 只有前两份进入清洗数据
            cleaned = [
                {
                    "source_path": "分析力.pdf",
                    "text": "分析相关内容",
                },
                {
                    "metadata": {
                        "source": "行动力.docx"
                    },
                    "text": "行动相关内容",
                },
            ]

            path = (
                root / "data" / "books_clean.json"
            )

            path.write_text(
                json.dumps(
                    cleaned,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = audit_knowledge(
                root,
                client_factory=(
                    lambda path: FakeChroma()
                ),
            )

            self.assertEqual(
                result["books"]["file_count"],
                3,
            )

            self.assertEqual(
                result["cleaned"]["record_count"],
                2,
            )

            self.assertEqual(
                result["comparison"][
                    "books_not_in_cleaned"
                ],
                ["价值观.pdf"],
            )

            self.assertEqual(
                result["comparison"][
                    "cleaned_not_in_chroma"
                ],
                [],
            )

    # ----------------------------------------
    # 测试5：未知JSON结构不能猜覆盖率
    # ----------------------------------------

    def test_unknown_schema(self):

        with tempfile.TemporaryDirectory() as temp:

            root = Path(temp)

            data = root / "data"
            data.mkdir()

            (data / "books_clean.json").write_text(
                '{"unknown": 1}',
                encoding="utf-8",
            )

            result = audit_knowledge(root)

            self.assertEqual(
                result["cleaned"]["status"],
                "unknown_schema",
            )

            self.assertIsNone(
                result["comparison"][
                    "books_not_in_cleaned"
                ]
            )


if __name__ == "__main__":
    unittest.main()
