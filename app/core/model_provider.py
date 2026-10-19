
"""
统一模型接口。

支持 OpenAI-compatible Chat Completions API。
使用者自行选择模型服务，不强制安装 Ollama。
"""

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass


@dataclass
class ModelConfig:
    base_url: str
    model_name: str
    api_key: str = ""
    timeout: int = 60


class ModelProvider:
    def __init__(self, config: ModelConfig):
        self.config = config

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        if not self.config.base_url.strip():
            raise ValueError("未配置模型 API 地址")

        if not self.config.model_name.strip():
            raise ValueError("未配置模型名称")

        url = (
            self.config.base_url.rstrip("/")
            + "/chat/completions"
        )

        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0,
        }

        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = (
                f"Bearer {self.config.api_key}"
            )

        request = urllib.request.Request(
            url,
            data=json.dumps(
                payload, ensure_ascii=False
            ).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout,
            ) as response:
                result = json.load(response)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"模型连接失败：{exc}"
            ) from exc

        try:
            return result["choices"][0][
                "message"
            ]["content"]
        except (
            KeyError, IndexError, TypeError
        ) as exc:
            raise RuntimeError(
                "模型返回格式不符合预期"
            ) from exc


def provider_from_env():
    config = ModelConfig(
        base_url=os.getenv(
            "MODEL_BASE_URL", ""
        ),
        model_name=os.getenv(
            "MODEL_NAME", ""
        ),
        api_key=os.getenv(
            "MODEL_API_KEY", ""
        ),
    )

    return ModelProvider(config)
