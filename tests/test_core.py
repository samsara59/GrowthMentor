
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from app.core.exceptions import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_int(
    environ: Mapping[str, str],
    key: str,
    default: int,
    minimum: int,
) -> int:

    raw = environ.get(key, str(default))

    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{key}必须是整数，实际值为：{raw}"
        ) from exc

    if value < minimum:
        raise ConfigurationError(
            f"{key}不能小于{minimum}"
        )

    return value


@dataclass(frozen=True)
class AppConfig:

    project_root: Path
    data_dir: Path

    memory_db: Path
    knowledge_db: Path
    vector_dir: Path

    chat_model: str
    embedding_model: str

    num_gpu: int
    timeout_seconds: int

    ollama_url: str

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "AppConfig":

        env = os.environ if environ is None else environ

        data_dir = PROJECT_ROOT / "data"

        chat_model = env.get(
            "GM_CHAT_MODEL",
            "gemma3:4b",
        ).strip()

        embedding_model = env.get(
            "GM_EMBEDDING_MODEL",
            "bge-m3",
        ).strip()

        if not chat_model:
            raise ConfigurationError(
                "GM_CHAT_MODEL不能为空"
            )

        if not embedding_model:
            raise ConfigurationError(
                "GM_EMBEDDING_MODEL不能为空"
            )

        num_gpu = read_int(
            env,
            "GM_NUM_GPU",
            default=0,
            minimum=0,
        )

        timeout_seconds = read_int(
            env,
            "GM_TIMEOUT",
            default=300,
            minimum=1,
        )

        ollama_url = env.get(
            "GM_OLLAMA_URL",
            "http://127.0.0.1:11434",
        ).strip().rstrip("/")

        if not ollama_url.startswith(
            ("http://", "https://")
        ):
            raise ConfigurationError(
                "GM_OLLAMA_URL必须使用http或https"
            )

        return cls(
            project_root=PROJECT_ROOT,
            data_dir=data_dir,
            memory_db=data_dir / "user_memory.db",
            knowledge_db=data_dir / "knowledge.db",
            vector_dir=data_dir / "chroma",
            chat_model=chat_model,
            embedding_model=embedding_model,
            num_gpu=num_gpu,
            timeout_seconds=timeout_seconds,
            ollama_url=ollama_url,
        )


def get_config() -> AppConfig:
    return AppConfig.from_env()
