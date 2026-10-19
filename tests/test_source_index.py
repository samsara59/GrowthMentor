
import unittest

from scripts.build_source_index import (
    extract_text,
    extract_location,
    split_text,
    iter_entries,
    preflight,
)


class TestSourceIndex(unittest.TestCase):

    def test_extract_text(self):

        record = {
            "source": "分析力.docx",
            "text": "这是一段测试教材文本。",
        }

        self.assertEqual(
            extract_text(record),
            "这是一段测试教材文本。",
        )

    def test_location(self):

        record = {
            "source": "分析力.docx",
            "page": 12,
        }

        self.assertEqual(
            extract_location(record),
            "page=12",
        )

    def test_chunk_overlap(self):

        text = "A" * 900

        chunks = list(
            split_text(
                text,
                chunk_size=800,
                overlap=120,
            )
        )

        self.assertEqual(len(chunks), 2)

        self.assertEqual(
            chunks[0][-120:],
            chunks[1][:120],
        )

    def test_missing_source_is_not_indexed(self):

        records = [
            {
                "text": "这是没有来源的信息。" * 20,
            }
        ]

        entries = list(
            iter_entries(records)
        )

        self.assertEqual(entries, [])

    def test_stable_ids_and_metadata(self):

        records = [
            {
                "source": "分析力.docx",
                "location": "第一章",
                "text": "这是教材正文。" * 100,
            }
        ]

        first = list(
            iter_entries(records)
        )

        second = list(
            iter_entries(records)
        )

        self.assertTrue(first)

        self.assertEqual(
            first[0][0],
            second[0][0],
        )

        self.assertEqual(
            first[0][2]["source"],
            "分析力.docx",
        )

    def test_preflight_counts(self):

        records = [
            {
                "source": "行动力.pdf",
                "text": "行动相关教材内容。" * 10,
            },
            {
                "text": "没有来源的文本。" * 10,
            },
        ]

        self.assertEqual(
            preflight(records),
            1,
        )


if __name__ == "__main__":
    unittest.main()
