
from pathlib import Path
import olefile


ROOT = Path(__file__).resolve().parents[1]

path = ROOT / "books" / (
    "24王爽：职来职爽答疑专栏.docx"
)

if not path.is_file():
    raise FileNotFoundError(path)

print("文件：", path.name)
print("是否为 OLE 文件：", olefile.isOleFile(path))

if not olefile.isOleFile(path):
    raise ValueError("文件不是有效的 OLE 复合文件")

with olefile.OleFileIO(path) as ole:

    # 列出内部数据流
    streams = [
        "/".join(parts)
        for parts in ole.listdir(
            streams=True,
            storages=False
        )
    ]

    print("\n内部数据流：")

    for stream in streams:
        print("  ", stream)

    # 判断可能的实际格式
    names = set(streams)

    if "EncryptedPackage" in names:
        print("\n可能是加密的 Office 文档")

    elif "WordDocument" in names:
        print("\n可能是旧版 Word DOC 格式")

    else:
        print("\n未识别为常见 Word 文档结构")
