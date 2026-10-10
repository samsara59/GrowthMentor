
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from app.ingestion.schema import DocumentRecord
from scripts.ingest_books import ingest_directory


class TestSourceMapping(unittest.TestCase):

    def test_read_mapped_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            original = root / "original.pdf"
            repaired = root / "repaired.pdf"

            original.write_bytes(b"original bytes")
            repaired.write_bytes(b"repaired bytes")

            source_map = {
                "original.pdf": {
                    "read_path": str(repaired),
                    "operation": "repair",
                    "warnings": ["原始文件存在问题"]
                }
            }

            fake_record = DocumentRecord(
                source="repaired.pdf",
                location="第1页",
                text="学习力教材正文",
                method="pdf_text"
            )

            with patch(
                "scripts.ingest_books.load_document",
                return_value=[fake_record]
            ) as mock_loader:

                records, report = ingest_directory(
                    root,
                    filename="original.pdf",
                    source_map=source_map
                )

                mock_loader.assert_called_once_with(
                    repaired
                )

            # 保存的来源必须是原始文件
            self.assertEqual(
                records[0]["source"],
                "original.pdf"
            )

            # 正文、页码和提取方式不能改变
            self.assertEqual(
                records[0]["text"],
                "学习力教材正文"
            )
            self.assertEqual(
                records[0]["location"],
                "第1页"
            )

            # 修复警告必须保留
            self.assertEqual(
                report[0]["status"],
                "needs_review"
            )
            self.assertEqual(
                report[0]["provenance"]["operation"],
                "repair"
            )
            self.assertIn(
                "原始文件存在问题",
                report[0]["provenance"]["warnings"]
            )

            # 原始文件不能被修改
            self.assertEqual(
                original.read_bytes(),
                b"original bytes"
            )

    def test_missing_mapped_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            original = root / "original.pdf"
            original.write_bytes(b"original bytes")

            missing = root / "missing.pdf"

            source_map = {
                "original.pdf": {
                    "read_path": str(missing),
                    "operation": "repair",
                    "warnings": ["修复副本尚未生成"]
                }
            }

            with patch(
                "scripts.ingest_books.load_document"
            ) as mock_loader:

                records, report = ingest_directory(
                    root,
                    filename="original.pdf",
                    source_map=source_map
                )

                # 不允许静默退回原文件
                mock_loader.assert_not_called()

            self.assertEqual(records, [])
            self.assertEqual(
                report[0]["status"],
                "error"
            )
            self.assertIn(
                "映射副本不存在",
                report[0]["error"]
            )


if __name__ == "__main__":
    unittest.main()
