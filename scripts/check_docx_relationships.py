
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]

path = ROOT / "books" / (
    "17 崔璀：分析力发挥（下），"
    "不擅交际也能实现高效合作.docx"
)

with ZipFile(path) as archive:

    relationship_files = [
        name
        for name in archive.namelist()
        if name.endswith(".rels")
    ]

    print("关系文件数量：", len(relationship_files))

    found = False

    for filename in relationship_files:

        root = ET.fromstring(
            archive.read(filename)
        )

        for relationship in root:

            target = relationship.attrib.get(
                "Target", ""
            )

            if "null" in target.lower():
                found = True

                print("\n发现可疑引用")
                print("关系文件：", filename)
                print(
                    "关系ID：",
                    relationship.attrib.get("Id")
                )
                print(
                    "关系类型：",
                    relationship.attrib.get("Type")
                )
                print("引用目标：", target)
                print(
                    "目标模式：",
                    relationship.attrib.get("TargetMode")
                )

    if not found:
        print("未发现 Target 中包含 NULL 的引用")

    print("\n检查正文是否使用 rId6")

    with ZipFile(path) as archive:
        root = ET.fromstring(
            archive.read("word/document.xml")
        )

        references = []

        for element in root.iter():
            for attribute, value in element.attrib.items():
                if value == "rId6":
                    references.append({
                        "element": element.tag.split("}")[-1],
                        "attribute": attribute.split("}")[-1]
                    })

        print("引用次数：", len(references))

        for item in references:
            print(item)