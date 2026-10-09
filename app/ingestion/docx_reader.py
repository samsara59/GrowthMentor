
from pathlib import Path
from docx import Document

from app.ingestion.schema import DocumentRecord


def read_docx(file_path: str | Path) -> list[DocumentRecord]:
    """读取 Word 文档的段落文字。"""

    # 将输入转换为 Path 对象
    path = Path(file_path)

    # 检查文件是否存在
    if not path.is_file():
        raise FileNotFoundError(
            f"Word 文件不存在: {path}"
        )

    # 打开 Word 文档
    doc = Document(path)

    # 保存文本记录
    records = []

    # 遍历全部段落
    for index, paragraph in enumerate(doc.paragraphs, 1):

        text = paragraph.text.strip()

        # 跳过空段落
        if not text:
            continue

        # 转换成统一数据结构
        record = DocumentRecord(
            source=path.name,
            location=f"第{index}段",
            text=text,
            method="docx_text"
        )

        records.append(record)

    return records
