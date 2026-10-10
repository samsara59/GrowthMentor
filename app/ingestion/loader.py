
from pathlib import Path

from app.ingestion.schema import DocumentRecord
from app.ingestion.pdf_reader import read_pdf
from app.ingestion.docx_reader import read_docx
from app.ingestion.image_reader import read_image


def load_document(
    file_path: str | Path,
    ocr_fallback: bool = False
) -> list[DocumentRecord]:
    """根据教材文件类型，调用相应的读取器。"""

    # 1. 转换为 Path 对象
    path = Path(file_path)

    # 2. 检查文件
    if not path.is_file():
        raise FileNotFoundError(
            f"教材文件不存在或不是普通文件: {path}"
        )

    # 3. 获取扩展名并转换成小写
    suffix = path.suffix.lower()

    # 4. 根据扩展名选择读取器
    readers = {
        ".pdf": read_pdf,
        ".docx": read_docx,
        ".png": read_image,
        ".jpg": read_image,
        ".jpeg": read_image,
    }

    # 5. 检查是否支持当前文件格式
    if suffix not in readers:
        raise ValueError(
            f"不支持的教材格式: {suffix}"
        )

    # 6. 调用对应的读取器
    reader = readers[suffix]

    # 只有 PDF 且明确开启 OCR 时，才传递额外参数
    if suffix == ".pdf" and ocr_fallback:
        return reader(path, ocr_fallback=True)

    return reader(path)
