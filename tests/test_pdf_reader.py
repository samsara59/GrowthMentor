
import unittest
import tempfile
from pathlib import Path

import pymupdf

from app.ingestion.pdf_reader import read_pdf


class TestPDFReader(unittest.TestCase):

    def test_read_pdf(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            pdf_path = Path(temp_dir) / "demo.pdf"

            # 创建一个两页 PDF
            with pymupdf.open() as pdf:

                page = pdf.new_page()
                page.insert_text(
                    (72, 72),
                    "Analytical strength test."
                )

                # 第二页故意留空
                pdf.new_page()

                pdf.save(pdf_path)

            # 执行我们刚编写的读取函数
            records = read_pdf(pdf_path)

            # 应当只读到一个非空页面
            self.assertEqual(len(records), 1)

            self.assertEqual(
                records[0].source,
                "demo.pdf"
            )

            self.assertEqual(
                records[0].location,
                "第1页"
            )

            self.assertEqual(
                records[0].method,
                "pdf_text"
            )

            self.assertIn(
                "Analytical strength",
                records[0].text
            )

    def test_missing_file(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.pdf"

            with self.assertRaises(FileNotFoundError):
                read_pdf(missing_path)


if __name__ == "__main__":
    unittest.main()
