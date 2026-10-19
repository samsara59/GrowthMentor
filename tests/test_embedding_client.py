
import json
import unittest

from unittest.mock import patch
from urllib.error import URLError

from app.rag.embedding_client import (
    OllamaEmbeddingClient,
)


class TestEmbeddingClient(unittest.TestCase):

    @patch("app.rag.embedding_client.urlopen")
    def test_batch_embedding(self, mock_urlopen):
        response = {
            "embeddings": [
                [0.6, 0.8],
                [0.8, 0.6],
            ]
        }

        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            json.dumps(response).encode("utf-8")
        )

        client = OllamaEmbeddingClient()

        vectors = client.embed([
            "学习方法",
            "如何高效阅读",
        ])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 2)

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))

        self.assertEqual(payload["model"], "bge-m3")
        self.assertEqual(payload["options"]["num_gpu"], 0)
        self.assertFalse(payload["truncate"])

    def test_empty_input(self):
        client = OllamaEmbeddingClient()

        with self.assertRaises(ValueError):
            client.embed([""])

    @patch("app.rag.embedding_client.urlopen")
    def test_wrong_vector_count(self, mock_urlopen):
        response = {
            "embeddings": [[0.1, 0.2]]
        }

        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            json.dumps(response).encode("utf-8")
        )

        client = OllamaEmbeddingClient()

        with self.assertRaises(RuntimeError):
            client.embed(["第一条", "第二条"])

    @patch("app.rag.embedding_client.urlopen")
    def test_connection_error(self, mock_urlopen):
        mock_urlopen.side_effect = URLError(
            "连接失败"
        )

        client = OllamaEmbeddingClient()

        with self.assertRaises(RuntimeError):
            client.embed(["学习力"])


if __name__ == "__main__":
    unittest.main()
