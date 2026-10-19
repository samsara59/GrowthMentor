
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image


# 1. 定位项目根目录
ROOT = Path(__file__).resolve().parents[1]

# 2. 指定测试图片
image_path = ROOT / "books" / "学习力1.jpg"

if not image_path.is_file():
    raise FileNotFoundError(
        f"测试图片不存在: {image_path}"
    )

# 3. 初始化中文 OCR 模型
reader = easyocr.Reader(
    ["ch_sim", "en"],
    gpu=False
)

# 4. 加载图片并转换为数组
with Image.open(image_path) as img:
    image_array = np.array(
        img.convert("RGB")
    )

# 5. 执行 OCR
results = reader.readtext(
    image_array,
    detail=1,
    paragraph=False
)

# 6. 输出识别结果
print(f"识别到 {len(results)} 个文字区域")

for index, (box, text, confidence) in enumerate(
    results, start=1
):
    print(f"\n第 {index} 个文字区域")
    print("识别文字：", text)
    print("置信度：", round(confidence, 3))
    print("文字坐标：", box)
