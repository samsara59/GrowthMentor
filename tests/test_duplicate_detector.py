
import tempfile
import unittest

from pathlib import Path

from app.ingestion.duplicate_detector import (
    compute_file_hash,
    find_duplicate_files,
)


class TestDuplicateDetector(unittest.TestCase):

    def test_same_content_same_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            a = root / "a.pdf"
            b = root / "b.pdf"

            a.write_bytes(b"same document")
            b.write_bytes(b"same document")

            self.assertEqual(
                compute_file_hash(a),
                compute_file_hash(b)
            )

    def test_detect_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            (root / "a.jpg").write_bytes(b"image one")
            (root / "b.jpg").write_bytes(b"image one")
            (root / "c.jpg").write_bytes(b"image two")

            groups = find_duplicate_files(root)

            self.assertEqual(
                groups,
                [["a.jpg", "b.jpg"]]
            )

    def test_do_not_modify_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            a = root / "a.png"
            b = root / "b.png"

            a.write_bytes(b"original data")
            b.write_bytes(b"original data")

            find_duplicate_files(root)

            self.assertEqual(
                a.read_bytes(),
                b"original data"
            )
            self.assertTrue(b.is_file())


if __name__ == "__main__":
    unittest.main()
