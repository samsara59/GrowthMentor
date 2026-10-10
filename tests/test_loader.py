
import unittest
import tempfile

from pathlib import Path
from unittest.mock import patch

from app.ingestion.loader import load_document
from app.ingestion.schema import DocumentRecord


class TestLoader(unittest.TestCase):

    def test_pdf_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.pdf"
            path.write_bytes(b"test")

            record = DocumentRecord(
                source="demo.pdf",
                location="第1页",
                text="测试内容",
                method="pdf_text"
            )

            with patch(
                "app.ingestion.loader.read_pdf"
            ) as mock_reader:
                mock_reader.return_value = [record]

                result = load_document(path)

                self.assertEqual(result, [record])
                mock_reader.assert_called_once_with(path)

    def test_docx_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.docx"
            path.write_bytes(b"test")

            with patch(
                "app.ingestion.loader.read_docx"
            ) as mock_reader:
                mock_reader.return_value = []

                load_document(path)

                mock_reader.assert_called_once_with(path)

    def test_image_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.PNG"
            path.write_bytes(b"test")

            with patch(
                "app.ingestion.loader.read_image"
            ) as mock_reader:
                mock_reader.return_value = []

                load_document(path)

                mock_reader.assert_called_once_with(path)

    def test_unsupported_format(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.txt"
            path.write_text("测试")

            with self.assertRaises(ValueError):
                load_document(path)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.pdf"

            with self.assertRaises(FileNotFoundError):
                load_document(path)


    def test_pdf_ocr_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scanned.PDF"
            path.write_bytes(b"test")

            with patch(
                "app.ingestion.loader.read_pdf"
            ) as mock_reader:

                mock_reader.return_value = []

                result = load_document(
                    path,
                    ocr_fallback=True
                )

                self.assertEqual(result, [])

                mock_reader.assert_called_once_with(
                    path,
                    ocr_fallback=True
                )

if __name__ == "__main__":
    unittest.main()
