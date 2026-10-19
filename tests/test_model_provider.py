
"""
Test the optional model provider without network.
"""

import json
import unittest
from unittest.mock import patch

from app.core.model_provider import (
    ModelConfig,
    ModelProvider,
)


class FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({
            "choices": [
                {
                    "message": {
                        "content": "测试成功"
                    }
                }
            ]
        }).encode("utf-8")


class TestModelProvider(unittest.TestCase):

    def setUp(self):
        self.provider = ModelProvider(
            ModelConfig(
                base_url="http://localhost:8000/v1",
                model_name="test-model",
                api_key="test-key",
            )
        )

    @patch("urllib.request.urlopen")
    def test_chat(self, mock_open):
        mock_open.return_value = FakeHTTPResponse()

        result = self.provider.chat(
            "你是助手",
            "你好",
        )

        self.assertEqual(result, "测试成功")

        request = mock_open.call_args.args[0]

        self.assertEqual(
            request.full_url,
            "http://localhost:8000/v1/chat/completions",
        )

        payload = json.loads(
            request.data.decode("utf-8")
        )

        self.assertEqual(
            payload["model"],
            "test-model",
        )

    def test_missing_model(self):
        provider = ModelProvider(
            ModelConfig(
                base_url="http://localhost:8000/v1",
                model_name="",
            )
        )

        with self.assertRaises(ValueError):
            provider.chat("system", "hello")


if __name__ == "__main__":
    unittest.main()
