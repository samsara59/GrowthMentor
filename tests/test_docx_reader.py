
import unittest
import tempfile
from pathlib import Path

from docx import Document

from app.ingestion.docx_reader import read_docx


class TestDOCXReader(unittest.TestCase):

    def test_read_docx(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            path = Path(temp_dir) / "demo.docx"

            # 创建测试 Word
            doc = Document()
            doc.add_paragraph("分析力优势")
            doc.add_paragraph("")
            doc.add_paragraph("喜欢分析问题和寻找原因。")
            doc.save(path)

            # 调用读取器
            records = read_docx(path)

            # 验证结果
            self.assertEqual(len(records), 2)

            self.assertEqual(
                records[0].text,
                "分析力优势"
            )

            self.assertEqual(
                records[1].location,
                "第3段"
            )

            self.assertEqual(
                records[0].method,
                "docx_text"
            )

    def test_missing_file(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            path = Path(temp_dir) / "missing.docx"

            with self.assertRaises(FileNotFoundError):
                read_docx(path)


if __name__ == "__main__":
    unittest.main()
