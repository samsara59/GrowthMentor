
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.ingestion.schema import DocumentRecord


# 1. 延迟加载并缓存 OCR 模型
@lru_cache(maxsize=1)
def get_ocr_reader():
    import easyocr

    return easyocr.Reader(
        ["ch_sim", "en"],
        gpu=False
    )


# 2. 正式图片读取函数
def read_image(
    file_path: str | Path
) -> list[DocumentRecord]:

    path = Path(file_path)

    # 检查文件是否存在
    if not path.is_file():
        raise FileNotFoundError(
            f"图片不存在: {path}"
        )

    # 读取图像，并转换为 RGB 数组
    try:
        with Image.open(path) as image:
            image_array = np.asarray(
                image.convert("RGB")
            )

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            f"无法解析图片: {path}"
        ) from exc

    # 调用 OCR 模型
    reader = get_ocr_reader()

    results = reader.readtext(
        image_array,
        detail=1,
        paragraph=False
    )

    # 提取识别到的文字
    lines = []

    for box, text, confidence in results:
        text = text.strip()

        if text:
            lines.append(text)

    # 没有识别到文字，返回空列表
    if not lines:
        return []

    # 当前版本先逐行拼接
    full_text = "\n".join(lines)

    # 转换为统一数据结构
    record = DocumentRecord(
        source=path.name,
        location="整张图片",
        text=full_text,
        method="ocr"
    )

    return [record]
