
import tempfile
import unittest
from pathlib import Path

import pymupdf
from docx import Document

from app.ingestion.file_detector import detect_file_format


class TestFileDetector(unittest.TestCase):

    def test_docx(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.docx"

            doc = Document()
            doc.add_paragraph("学习力")
            doc.save(path)

            info = detect_file_format(path)

            self.assertEqual(info.detected_format, "docx")
            self.assertTrue(info.extension_matches)

    def test_pdf(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.pdf"

            with pymupdf.open() as pdf:
                pdf.new_page()
                pdf.save(path)

            info = detect_file_format(path)

            self.assertEqual(info.detected_format, "pdf")
            self.assertTrue(info.extension_matches)

    def test_wrong_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fake.docx"

            # 文件名是 DOCX，实际内容却是 PDF
            with pymupdf.open() as pdf:
                pdf.new_page()
                pdf.save(path)

            info = detect_file_format(path)

            self.assertEqual(info.detected_format, "pdf")
            self.assertFalse(info.extension_matches)

    def test_unknown_format(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unknown.docx"
            path.write_bytes(b"not a real document")

            info = detect_file_format(path)

            self.assertEqual(info.detected_format, "unknown")
            self.assertFalse(info.extension_matches)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.docx"

            with self.assertRaises(FileNotFoundError):
                detect_file_format(path)


if __name__ == "__main__":
    unittest.main()
