
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import olefile


@dataclass(frozen=True)
class FileFormatInfo:
    filename: str
    extension: str
    detected_format: str
    extension_matches: bool


EXPECTED_FORMATS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
}


def detect_file_format(
    file_path: str | Path
) -> FileFormatInfo:

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    # 1. 读取文件头
    with path.open("rb") as f:
        header = f.read(8)

    detected = "unknown"

    # 2. 常见文件签名
    if header.startswith(b"%PDF-"):
        detected = "pdf"

    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        detected = "png"

    elif header.startswith(b"\xff\xd8\xff"):
        detected = "jpeg"

    # 3. 判断是否为标准 DOCX
    elif is_zipfile(path):

        with ZipFile(path) as archive:
            names = set(archive.namelist())

        if (
            "[Content_Types].xml" in names
            and "word/document.xml" in names
        ):
            detected = "docx"
        else:
            detected = "other_zip"

    # 4. 判断旧版 Word / Office 复合文件
    elif olefile.isOleFile(path):

        with olefile.OleFileIO(path) as ole:

            if ole.exists("EncryptedPackage"):
                detected = "encrypted_office"

            elif ole.exists("WordDocument"):
                detected = "doc"

            else:
                detected = "other_ole"

    # 5. 检查扩展名是否匹配
    extension = path.suffix.lower()

    expected = EXPECTED_FORMATS.get(extension)

    return FileFormatInfo(
        filename=path.name,
        extension=extension,
        detected_format=detected,
        extension_matches=(expected == detected)
    )
