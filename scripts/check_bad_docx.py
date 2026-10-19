
from pathlib import Path
from zipfile import ZipFile, is_zipfile


ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "17 崔璀：分析力发挥（下），不擅交际也能实现高效合作.docx",
    "24王爽：职来职爽答疑专栏.docx",
]


def inspect_docx(filename: str):
    path = ROOT / "books" / filename

    print(f"\n检查文件：{filename}")

    if not path.is_file():
        print("文件不存在")
        return

    print("文件大小：", path.stat().st_size, "字节")

    with path.open("rb") as f:
        header = f.read(8)

    print("文件头：", header.hex(" "))
    print("是否为有效 ZIP：", is_zipfile(path))

    if not is_zipfile(path):
        print("无法按标准 DOCX 结构读取")
        return

    try:
        with ZipFile(path) as archive:
            names = archive.namelist()

            print("内部文件数：", len(names))
            print(
                "包含 document.xml：",
                "word/document.xml" in names
            )

            bad_member = archive.testzip()

            if bad_member is None:
                print("ZIP 文件完整性检查：通过")
            else:
                print("ZIP 文件异常成员：", bad_member)

            null_targets = [
                name for name in names
                if name == "NULL" or name.endswith("/NULL")
            ]

            print("名称为 NULL 的成员：", null_targets)

    except Exception as exc:
        print("ZIP 检查失败：", type(exc).__name__)
        print("详细原因：", exc)


for filename in FILES:
    inspect_docx(filename)
