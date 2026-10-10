
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ingestion.schema import DocumentRecord
from scripts.ingest_books import ingest_directory
from unittest.mock import patch

class TestIngestBooks(unittest.TestCase):

    def test_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "demo.pdf").write_bytes(b"test")

            record = DocumentRecord(
                source="demo.pdf",
                location="第1页",
                text="分析力教材正文。" * 30,
                method="pdf_text"
            )

            with patch(
                "scripts.ingest_books.load_document",
                return_value=[record]
            ):
                records, report = ingest_directory(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(report[0]["status"], "prelim_ok")

    def test_failure_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            (root / "a.pdf").write_bytes(b"broken")
            (root / "b.pdf").write_bytes(b"valid")

            record = DocumentRecord(
                source="b.pdf",
                location="第1页",
                text="学习力教材正文。",
                method="pdf_text"
            )

            with patch(
                "scripts.ingest_books.load_document",
                side_effect=[
                    RuntimeError("模拟读取失败"),
                    [record]
                ]
            ):
                records, report = ingest_directory(root)

            self.assertEqual(report[0]["status"], "error")
            self.assertEqual(report[1]["status"], "needs_review")
            self.assertEqual(len(records), 1)

    def test_single_pdf_with_ocr(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            # 创建两个文件
            target = root / "scan.pdf"
            other = root / "other.pdf"

            target.write_bytes(b"test")
            other.write_bytes(b"test")

            # 模拟读取器
            with patch(
                "scripts.ingest_books.load_document",
                return_value=[]
            ) as mock_reader:

                records, report = ingest_directory(
                    root,
                    filename="scan.pdf",
                    ocr_fallback=True
                )

            # 只处理指定的一份教材
            self.assertEqual(len(report), 1)
            self.assertEqual(report[0]["file"], "scan.pdf")

            # 验证 OCR 参数正确传递
            mock_reader.assert_called_once_with(
                target,
                ocr_fallback=True
            )

    def test_wrong_docx_format(self):
        with tempfile.TemporaryDirectory() as folder:

            root = Path(folder)
            path = root / "fake.docx"

            # 模拟错误扩展名的 Word 文件
            path.write_bytes(b"fake word document")

            with patch(
                "scripts.ingest_books.detect_file_format"
            ) as mock_detector:

                with patch(
                    "scripts.ingest_books.load_document"
                ) as mock_loader:

                    mock_detector.return_value.extension = ".docx"
                    mock_detector.return_value.detected_format = "doc"
                    mock_detector.return_value.extension_matches = False

                    records, report = ingest_directory(root)

                    # 格式异常，不应调用正文读取器
                    mock_loader.assert_not_called()

            self.assertEqual(records, [])
            self.assertEqual(len(report), 1)

            self.assertEqual(
                report[0]["status"],
                "needs_conversion"
            )

            self.assertEqual(
                report[0]["detected_format"],
                "doc"
            )


if __name__ == "__main__":
    unittest.main()
