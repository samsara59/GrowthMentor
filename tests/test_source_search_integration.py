import tempfile
import unittest
from pathlib import Path

from app.capabilities.source_search import (
    verify_retrieved_sources,
)


class TestSourceSearchIntegration(unittest.TestCase):

    def test_matching_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            text = (
                "Practice may improve self regulation "
                "when repeated over a long period of time."
            )
            (root / "book.txt").write_text(
                text, encoding="utf-8"
            )

            result = verify_retrieved_sources(
                [{
                    "source": "book.txt",
                    "quote": text,
                    "verification": "retrieved_not_checked",
                }],
                root,
            )

            self.assertEqual(
                result[0]["verification"],
                "verified_against_original",
            )
            self.assertEqual(
                result[0]["original_text"],
                text,
            )

    def test_missing_original(self):
        with tempfile.TemporaryDirectory() as folder:
            result = verify_retrieved_sources(
                [{
                    "source": "missing.txt",
                    "quote": "A" * 60,
                }],
                Path(folder),
            )

            self.assertEqual(
                result[0]["verification"],
                "retrieved_not_checked",
            )
            self.assertNotIn(
                "original_text",
                result[0],
            )

    def test_false_quote(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "book.txt").write_text(
                "A" * 80, encoding="utf-8"
            )

            result = verify_retrieved_sources(
                [{
                    "source": "book.txt",
                    "quote": "B" * 80,
                }],
                root,
            )

            self.assertEqual(
                result[0]["verification"],
                "retrieved_not_checked",
            )

    def test_missing_books_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "missing"
            result = verify_retrieved_sources(
                [{
                    "source": "book.txt",
                    "quote": "A" * 60,
                    "verification": "retrieved_not_checked",
                }],
                missing,
            )

            self.assertEqual(
                result[0]["verification"],
                "retrieved_not_checked",
            )


if __name__ == "__main__":
    unittest.main()
