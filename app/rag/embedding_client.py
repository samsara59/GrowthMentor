
import json
import math

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class OllamaEmbeddingClient:
    """通过 Ollama 将文本转换为向量。"""

    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 300,
        num_gpu: int | None = 0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_gpu = num_gpu

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """批量生成文本向量。"""

        if not texts or any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError("输入必须是非空文本列表")

        options = {}

        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu

        payload = {
            "model": self.model,
            "input": texts,
            "truncate": False,
            "options": options,
        }

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        request = Request(
            url=f"{self.base_url}/api/embed",
            data=body,
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

        except HTTPError as exc:
            detail = exc.read().decode(
                "utf-8",
                errors="replace",
            )
            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {detail}"
            ) from exc

        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"无法连接 Ollama: {exc}"
            ) from exc

        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Ollama 返回的数据无法解析"
            ) from exc

        vectors = result.get("embeddings")

        if (
            not isinstance(vectors, list)
            or len(vectors) != len(texts)
            or not vectors
        ):
            raise RuntimeError("Embedding 返回数量不正确")

        dimension = len(vectors[0])

        if dimension == 0:
            raise RuntimeError("Embedding 向量不能为空")

        for vector in vectors:
            if (
                not isinstance(vector, list)
                or len(vector) != dimension
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in vector
                )
            ):
                raise RuntimeError("Embedding 向量数据无效")

        return vectors
