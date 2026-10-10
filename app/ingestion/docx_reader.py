
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.schema import DocumentRecord


def read_docx(file_path: str | Path) -> list[DocumentRecord]:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Word 文件不存在: {path}"
        )

    doc = Document(path)
    records = []

    paragraph_index = 0
    table_index = 0

    # 按 Word 中的实际顺序遍历段落和表格
    for block in doc.iter_inner_content():

        if isinstance(block, Paragraph):
            paragraph_index += 1
            text = block.text.strip()

            if text:
                records.append(
                    DocumentRecord(
                        source=path.name,
                        location=f"第{paragraph_index}段",
                        text=text,
                        method="docx_text"
                    )
                )

        elif isinstance(block, Table):
            table_index += 1

            for row_index, row in enumerate(
                block.rows, start=1
            ):
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                ]

                if not any(cells):
                    continue

                text = " | ".join(cells)

                records.append(
                    DocumentRecord(
                        source=path.name,
                        location=(
                            f"第{table_index}表"
                            f"第{row_index}行"
                        ),
                        text=text,
                        method="docx_text"
                    )
                )

    return records
