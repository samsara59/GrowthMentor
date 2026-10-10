
import unittest
import tempfile

from pathlib import Path
from unittest.mock import patch
from PIL import Image

from app.ingestion.image_reader import read_image


class TestImageReader(unittest.TestCase):

    def test_read_image(self):

        with tempfile.TemporaryDirectory() as folder:

            path = Path(folder) / "demo.png"

            # 生成一张测试图片
            Image.new(
                "RGB", (120, 60), "white"
            ).save(path)

            # 用 Mock 替代真实 OCR 模型
            with patch(
                "app.ingestion.image_reader.get_ocr_reader"
            ) as mock_reader:

                mock_reader.return_value.readtext.return_value = [
                    (
                        [[0, 0], [100, 0],
                         [100, 30], [0, 30]],
                        "学习力解析",
                        0.95
                    )
                ]

                records = read_image(path)

            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0].text,
                "学习力解析"
            )
            self.assertEqual(
                records[0].method,
                "ocr"
            )

    def test_empty_result(self):

        with tempfile.TemporaryDirectory() as folder:

            path = Path(folder) / "blank.png"

            Image.new(
                "RGB", (100, 100), "white"
            ).save(path)

            with patch(
                "app.ingestion.image_reader.get_ocr_reader"
            ) as mock_reader:

                mock_reader.return_value.readtext.return_value = []

                records = read_image(path)

            self.assertEqual(records, [])

    def test_missing_file(self):

        with tempfile.TemporaryDirectory() as folder:

            path = Path(folder) / "missing.png"

            with self.assertRaises(FileNotFoundError):
                read_image(path)

    def test_invalid_image(self):

        with tempfile.TemporaryDirectory() as folder:

            path = Path(folder) / "broken.png"

            # 伪造一份损坏的图片
            path.write_bytes(b"not a real image")

            with self.assertRaises(ValueError):
                read_image(path)


if __name__ == "__main__":
    unittest.main()
