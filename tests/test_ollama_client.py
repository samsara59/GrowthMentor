
import json
import unittest

from unittest.mock import patch
from urllib.error import URLError

from app.knowledge.ollama_client import OllamaClient


class TestOllamaClient(unittest.TestCase):

    @patch("app.knowledge.ollama_client.urlopen")
    def test_generate(self, mock_urlopen):

        # 模拟 Ollama 返回的数据
        fake_response = {
            "response": '[{"concept":"学习力"}]'
        }

        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            json.dumps(fake_response).encode("utf-8")
        )

        client = OllamaClient(model="test-model")

        result = client.generate("提取知识")

        self.assertEqual(
            result,
            '[{"concept":"学习力"}]'
        )

        # 检查程序实际发送的请求
        request = mock_urlopen.call_args.args[0]

        payload = json.loads(
            request.data.decode("utf-8")
        )

        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["prompt"], "提取知识")
        self.assertFalse(payload["stream"])

    @patch("app.knowledge.ollama_client.urlopen")
    def test_connection_error(self, mock_urlopen):

        mock_urlopen.side_effect = URLError(
            "连接被拒绝"
        )

        client = OllamaClient()

        with self.assertRaises(RuntimeError):
            client.generate("测试")

    @patch("app.knowledge.ollama_client.urlopen")
    def test_missing_response(self, mock_urlopen):

        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            b"{}"
        )

        client = OllamaClient()

        with self.assertRaises(RuntimeError):
            client.generate("测试")


if __name__ == "__main__":
    unittest.main()
