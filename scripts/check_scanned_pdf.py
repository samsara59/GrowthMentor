
from pathlib import Path

import numpy as np
import pymupdf

from app.ingestion.image_reader import get_ocr_reader


ROOT = Path(__file__).resolve().parents[1]
pdf_path = ROOT / "books" / "学习力.pdf"

if not pdf_path.is_file():
    raise FileNotFoundError(pdf_path)

# 1. 打开 PDF，只读取第一页
with pymupdf.open(pdf_path) as pdf:
    print("PDF 总页数：", len(pdf))

    page = pdf[0]

    # 2. 尝试直接提取文字
    original_text = page.get_text("text").strip()
    print("直接提取字符数：", len(original_text))

    # 3. 将 PDF 页面渲染为 RGB 图像
    pix = page.get_pixmap(
        dpi=150,
        colorspace=pymupdf.csRGB,
        alpha=False
    )

    image_array = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        3
    ).copy()

print("页面图像尺寸：", image_array.shape)

# 4. 使用已实现的模型缓存
reader = get_ocr_reader()

# 5. 对 PDF 页面进行 OCR
results = reader.readtext(
    image_array,
    detail=1,
    paragraph=False
)

print("OCR 识别区域数：", len(results))

# 6. 查看前 12 条识别结果
for index, (_, text, confidence) in enumerate(
    results[:12],
    start=1
):
    print(
        f"{index}. {text} "
        f"(置信度：{confidence:.3f})"
    )
