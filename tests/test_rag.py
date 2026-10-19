
from pathlib import Path
import pymupdf

from app.ingestion.schema import DocumentRecord


def read_pdf(file_path: str | Path) -> list[DocumentRecord]:
    """读取 PDF 中的文字，返回统一的教材记录。"""

    # 1. 转换为 Path 对象
    path = Path(file_path)

    # 2. 检查文件是否存在
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    # 3. 存放读取结果
    records = []
    empty_pages = []

    # 4. 打开 PDF
    with pymupdf.open(path) as pdf:

        # 5. 逐页读取
        for page_index, page in enumerate(pdf):

            text = page.get_text("text").strip()

            # 6. 没有提取到文字则记录页码
            if not text:
                empty_pages.append(page_index + 1)
                continue

            # 7. 转换为我们定义的数据结构
            record = DocumentRecord(
                source=path.name,
                location=f"第{page_index + 1}页",
                text=text,
                method="pdf_text"
            )

            records.append(record)

    # 8. 提醒可能需要 OCR 的页面
    if empty_pages:
        print(
            f"[提示] {path.name} 以下页面未提取到文字："
            f"{empty_pages}，可能需要 OCR。"
        )

    return records
