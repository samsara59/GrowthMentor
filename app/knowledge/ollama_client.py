
import json

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class OllamaClient:
    """通过 HTTP API 调用本地 Ollama 模型。"""

    def __init__(
        self,
        model: str = "gemma3:4b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 300,
        num_gpu: int | None = None,
        response_format: str | dict = "json",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_gpu = num_gpu
        self.response_format = response_format

    def generate(self, prompt: str) -> str:
        """发送 Prompt，返回模型生成的文本。"""

        # 1. 生成参数
        options = {
            "temperature": 0
        }

        # 当前电脑 GPU 存在 CUDA 兼容问题
        # num_gpu=0 时强制使用 CPU
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu

        # 2. HTTP 请求内容
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": self.response_format,
            "options": options,
        }

        # 3. Python 字典 -> JSON -> UTF-8 字节
        body = json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8")

        # 4. 构造 POST 请求
        request = Request(
            url=f"{self.base_url}/api/generate",
            data=body,
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        # 5. 发送请求并解析 Ollama 响应
        try:
            with urlopen(
                request,
                timeout=self.timeout
            ) as response:

                result = json.loads(
                    response.read().decode("utf-8")
                )

        except HTTPError as exc:

            detail = exc.read().decode(
                "utf-8",
                errors="replace"
            )

            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {detail}"
            ) from exc

        except (URLError, TimeoutError) as exc:

            raise RuntimeError(
                f"无法连接 Ollama 服务: {exc}"
            ) from exc

        except (
            UnicodeDecodeError,
            json.JSONDecodeError
        ) as exc:

            raise RuntimeError(
                "Ollama 返回的数据无法解析"
            ) from exc

        # 6. 提取生成文本
        output = result.get("response")

        if not isinstance(output, str):
            raise RuntimeError(
                "Ollama 响应缺少 response 字段"
            )

        return output
