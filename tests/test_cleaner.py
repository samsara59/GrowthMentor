
import unittest

from app.ingestion.cleaner import (
    clean_text,
    clean_records,
)


class TestCleaner(unittest.TestCase):

    def test_remove_navigation(self):
        text = "学习力\n返回顶部\n\n行动力"

        cleaned, removed = clean_text(text)

        self.assertEqual(
            cleaned,
            "学习力\n行动力"
        )
        self.assertEqual(removed, ["返回顶部"])

    def test_keep_short_title(self):
        cleaned, removed = clean_text("共情力")

        self.assertEqual(cleaned, "共情力")
        self.assertEqual(removed, [])

    def test_keep_meaningful_sentence(self):
        text = "返回顶部是一种网页操作。"

        cleaned, removed = clean_text(text)

        self.assertEqual(cleaned, text)
        self.assertEqual(removed, [])

    def test_preserve_original_record(self):
        records = [{
            "source": "学习力.docx",
            "location": "第3段",
            "text": "返回首页\n学习力  是一种优势。",
            "method": "docx_text"
        }]

        cleaned, report = clean_records(records)

        # 原始输入不能被修改
        self.assertEqual(
            records[0]["text"],
            "返回首页\n学习力  是一种优势。"
        )

        # 清洗后的结果
        self.assertEqual(
            cleaned[0]["text"],
            "学习力 是一种优势。"
        )

        # 来源必须保留
        self.assertEqual(
            cleaned[0]["location"],
            "第3段"
        )

        self.assertEqual(
            report[0]["removed_lines"],
            ["返回首页"]
        )

    def test_url_is_flagged_not_deleted(self):
        records = [{
            "source": "课程.pdf",
            "location": "第1页",
            "text": "https://example.com\n课程介绍",
            "method": "pdf_text"
        }]

        cleaned, report = clean_records(records)

        self.assertEqual(len(cleaned), 1)
        self.assertIn(
            "https://example.com",
            cleaned[0]["text"]
        )
        self.assertIn(
            "包含独立网址行",
            report[0]["flags"]
        )

    def test_empty_after_cleaning(self):
        records = [{
            "source": "导航页.docx",
            "location": "第1段",
            "text": "返回顶部",
            "method": "docx_text"
        }]

        cleaned, report = clean_records(records)

        self.assertEqual(cleaned, [])
        self.assertIn(
            "清洗后没有正文",
            report[0]["flags"]
        )


if __name__ == "__main__":
    unittest.main()
