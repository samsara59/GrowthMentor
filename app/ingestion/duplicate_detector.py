
import hashlib

from pathlib import Path
from collections import defaultdict


SUPPORTED = {
    ".pdf",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg"
}


def compute_file_hash(file_path: str | Path) -> str:
    """计算文件内容的 SHA-256。"""

    path = Path(file_path)

    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def find_duplicate_files(
    folder: str | Path
) -> list[list[str]]:
    """返回同一文件夹中的文件级重复组。"""

    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(folder)

    hash_groups = defaultdict(list)

    for path in sorted(folder.iterdir()):

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED:
            continue

        digest = compute_file_hash(path)

        hash_groups[digest].append(path.name)

    duplicate_groups = [
        filenames
        for filenames in hash_groups.values()
        if len(filenames) > 1
    ]

    return duplicate_groups
