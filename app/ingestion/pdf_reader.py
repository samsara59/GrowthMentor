
from pathlib import Path

import numpy as np
import pymupdf

from app.ingestion.schema import DocumentRecord
from app.ingestion.image_reader import get_ocr_reader


def ocr_pdf_page(page) -> str:
    """将一页 PDF 渲染为图片，并进行 OCR。"""

    # 1. PDF 页面转 RGB 图像
    pix = page.get_pixmap(
        dpi=150,
        colorspace=pymupdf.csRGB,
        alpha=False
    )

    # 2. 像素数据转为 NumPy 数组
    image_array = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        3
    ).copy()

    # 3. 复用已有的 OCR 模型
    reader = get_ocr_reader()

    results = reader.readtext(
        image_array,
        detail=1,
        paragraph=False
    )

    # 4. 提取非空文本
    lines = []

    for box, text, confidence in results:
        text = text.strip()

        if text:
            lines.append(text)

    return "\n".join(lines)


def read_pdf(
    file_path: str | Path,
    ocr_fallback: bool = False
) -> list[DocumentRecord]:
    """读取 PDF，可选择对无文字页面启用 OCR。"""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"文件不存在: {path}"
        )

    records = []
    empty_pages = []

    with pymupdf.open(path) as pdf:

        for page_index, page in enumerate(pdf):

            # 优先直接提取 PDF 文字
            text = page.get_text("text").strip()
            method = "pdf_text"

            # 常规提取失败时，按需执行 OCR
            if not text and ocr_fallback:
                text = ocr_pdf_page(page)
                method = "ocr"

            # 两种方法都没有获得文字
            if not text:
                empty_pages.append(page_index + 1)
                continue

            # 转为统一数据结构
            records.append(
                DocumentRecord(
                    source=path.name,
                    location=f"第{page_index + 1}页",
                    text=text,
                    method=method
                )
            )

    if empty_pages:
        print(
            f"[提示] {path.name} "
            f"以下页面未提取到文字：{empty_pages}"
        )

    return records
