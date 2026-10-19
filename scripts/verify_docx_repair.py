
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from app.ingestion.docx_reader import read_docx


ROOT = Path(__file__).resolve().parents[1]

original = ROOT / "books" / (
    "17 崔璀：分析力发挥（下），"
    "不擅交际也能实现高效合作.docx"
)

repaired = (
    ROOT / "data" / "repaired"
    / "17_分析力发挥下_repaired.docx"
)

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_xml_text(path):
    """直接获取 DOCX 主文档中的文字节点。"""

    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml")

    root = ET.fromstring(xml)

    return [
        node.text or ""
        for node in root.iter(W + "t")
    ]


# 比较修复前后的正文
original_text = extract_xml_text(original)
repaired_text = extract_xml_text(repaired)

print("原始文字节点数：", len(original_text))
print("修复后文字节点数：", len(repaired_text))

print(
    "主文档文字是否一致：",
    original_text == repaired_text
)

# 使用正式读取器验证修复副本
records = read_docx(repaired)

print("有效 DocumentRecord 数量：", len(records))

for record in records[:3]:
    print(record.location, repr(record.text[:100]))
