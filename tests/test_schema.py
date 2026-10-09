
import unittest

from app.ingestion.schema import DocumentRecord


class TestDocumentRecord(unittest.TestCase):

    def test_valid_record(self):
        record = DocumentRecord(
            source="demo.pdf",
            location="第1页",
            text="分析力需要结合真实行为验证。",
            method="pdf_text"
        )

        self.assertEqual(record.source, "demo.pdf")
        self.assertEqual(record.method, "pdf_text")

        result = record.to_dict()

        self.assertEqual(
            result["text"],
            "分析力需要结合真实行为验证。"
        )

    def test_empty_text(self):
        with self.assertRaises(ValueError):
            DocumentRecord(
                source="demo.png",
                location="图片正文",
                text="   ",
                method="ocr"
            )


if __name__ == "__main__":
    unittest.main()
