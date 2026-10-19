import json
import os
import unittest
from io import BytesIO
from unittest.mock import patch

from app.capabilities.source_search import (
    LocalSourceSearch,
)


class FakeResponse:
    def __init__(self, payload):
        self.stream = BytesIO(
            json.dumps(payload).encode("utf-8")
        )

    def __enter__(self):
        return self.stream

    def __exit__(self, *args):
        self.stream.close()


class TestEmbeddingBackend(unittest.TestCase):

    def setUp(self):
        self.backend = LocalSourceSearch(
            collection=object()
        )

    def test_openai_backend(self):
        env = {
            "GROWTH_EMBEDDING_BACKEND": "openai",
            "EMBEDDING_BASE_URL": "http://localhost:8000/v1",
            "EMBEDDING_MODEL_NAME": "test-embedding",
            "EMBEDDING_API_KEY": "test-key",
        }

        def fake_urlopen(request, timeout):
            self.assertEqual(
                request.full_url,
                "http://localhost:8000/v1/embeddings",
            )

            payload = json.loads(
                request.data.decode("utf-8")
            )

            self.assertEqual(
                payload["model"],
                "test-embedding",
            )

            self.assertEqual(
                payload["input"],
                "hello",
            )

            return FakeResponse({
                "data": [{
                    "embedding": [0.1, 0.2, 0.3],
                }]
            })

        with patch.dict(os.environ, env):
            with patch(
                "urllib.request.urlopen",
                side_effect=fake_urlopen,
            ):
                vector = self.backend._embed("hello")

        self.assertEqual(
            vector,
            [0.1, 0.2, 0.3],
        )

    def test_missing_config(self):
        env = {
            "GROWTH_EMBEDDING_BACKEND": "openai",
            "EMBEDDING_BASE_URL": "",
            "EMBEDDING_MODEL_NAME": "",
        }

        with patch.dict(os.environ, env):
            with self.assertRaises(ValueError):
                self.backend._embed("hello")

    def test_invalid_backend(self):
        with patch.dict(
            os.environ,
            {"GROWTH_EMBEDDING_BACKEND": "unknown"},
        ):
            with self.assertRaises(ValueError):
                self.backend._embed("hello")

    def test_invalid_response(self):
        env = {
            "GROWTH_EMBEDDING_BACKEND": "openai",
            "EMBEDDING_BASE_URL": "http://localhost/v1",
            "EMBEDDING_MODEL_NAME": "test",
        }

        with patch.dict(os.environ, env):
            with patch(
                "urllib.request.urlopen",
                return_value=FakeResponse({
                    "data": [{"embedding": []}],
                }),
            ):
                with self.assertRaises(RuntimeError):
                    self.backend._embed("hello")


if __name__ == "__main__":
    unittest.main()
