
import logging


LOGGER_NAME = "growthmentor"


def setup_logging(
    level: str = "INFO",
) -> logging.Logger:
    """
    配置项目统一日志。

    当前只输出到终端。
    不自动记录用户原始经历、
    对话内容或完整LLM Prompt。
    """

    numeric_level = getattr(
        logging,
        level.upper(),
        None,
    )

    if not isinstance(numeric_level, int):
        raise ValueError(
            f"无效的日志级别：{level}"
        )

    logger = logging.getLogger(
        LOGGER_NAME
    )

    logger.setLevel(numeric_level)

    # 防止Streamlit反复运行时重复添加Handler
    if not logger.handlers:

        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | "
                "%(levelname)s | "
                "%(name)s | "
                "%(message)s"
            )
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)

    logger.propagate = False

    return logger


def get_logger(
    component: str,
) -> logging.Logger:

    setup_logging()

    return logging.getLogger(
        f"{LOGGER_NAME}.{component}"
    )
