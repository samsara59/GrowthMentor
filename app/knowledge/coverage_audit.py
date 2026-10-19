
"""只读检查本地教材、SQLite与Chroma。"""

import json
import re
import sqlite3

from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Callable


# ============================================
# 1. 支持的教材格式和来源字段
# ============================================

SUFFIXES = {
    ".pdf", ".docx", ".doc",
    ".png", ".jpg", ".jpeg",
    ".txt", ".md",
}

SOURCE_KEYS = (
    "source_file",
    "source_path",
    "file_path",
    "filename",
    "file_name",
    "source",
    "path",
    "origin",
)

DATA_KEYS = (
    "records",
    "items",
    "documents",
    "chunks",
    "data",
    "pages",
)

FILE_PATTERN = re.compile(
    r"([^/\\]+?\.(?:pdf|docx|doc|png|jpg|jpeg|txt|md))"
    r"(?=$|[#:?\s])",
    re.IGNORECASE,
)


# ============================================
# 2. 从元数据中提取教材文件名
# ============================================

def filename_from_record(
    record: Any,
) -> str | None:

    if not isinstance(record, dict):
        return None

    # 同时适配顶层字段和metadata字段
    for obj in (
        record,
        record.get("metadata"),
    ):

        if not isinstance(obj, dict):
            continue

        for key in SOURCE_KEYS:

            value = obj.get(key)

            if not isinstance(value, str):
                continue

            # Windows路径和Unix路径统一处理
            normalized = value.replace("\\", "/")

            match = FILE_PATTERN.search(normalized)

            if match:
                return match.group(1)

    return None


# ============================================
# 3. 扫描books原始教材
# ============================================

def scan_books(folder: Path) -> dict:

    if not folder.is_dir():
        return {
            "status": "missing",
            "file_count": 0,
            "names": [],
            "duplicate_names": [],
        }

    names = [
        item.name
        for item in folder.rglob("*")
        if (
            item.is_file()
            and item.suffix.lower() in SUFFIXES
        )
    ]

    counts = Counter(
        name.casefold()
        for name in names
    )

    duplicates = [
        name
        for name, count in counts.items()
        if count > 1
    ]

    return {
        "status": "ok",
        "file_count": len(names),
        "names": sorted(set(names)),
        "duplicate_names": sorted(duplicates),
    }


# ============================================
# 4. 审计books_clean.json
# ============================================

def scan_cleaned(path: Path) -> dict:

    if not path.is_file():
        return {
            "status": "missing",
            "record_count": None,
            "names": [],
            "without_source": None,
            "sample_record_keys": [],
        }

    try:
        content = json.loads(
            path.read_text(encoding="utf-8-sig")
        )

    except (OSError, ValueError) as exc:
        return {
            "status": "error",
            "error": str(exc),
            "record_count": None,
            "names": [],
        }

    if isinstance(content, list):
        records = content

    elif isinstance(content, dict):
        records = next(
            (
                content[key]
                for key in DATA_KEYS
                if isinstance(content.get(key), list)
            ),
            None,
        )

    else:
        records = None

    if records is None:
        return {
            "status": "unknown_schema",
            "record_count": None,
            "names": [],
            "root_keys": (
                list(content.keys())
                if isinstance(content, dict)
                else []
            ),
        }

    names = []
    without_source = 0

    for record in records:

        filename = filename_from_record(record)

        if filename is None:
            without_source += 1
        else:
            names.append(filename)

    return {
        "status": "ok",
        "record_count": len(records),
        "names": sorted(set(names)),
        "without_source": without_source,
        "sample_record_keys": (
            list(records[0].keys())
            if (
                records
                and isinstance(records[0], dict)
            )
            else []
        ),
    }


# ============================================
# 5. 只读审计SQLite
# ============================================

def scan_sqlite(path: Path) -> dict:

    if not path.is_file():
        return {
            "status": "missing",
            "tables": [],
        }

    tables = []

    try:
        # mode=ro：只读打开数据库
        uri = path.resolve().as_uri() + "?mode=ro"

        with closing(
            sqlite3.connect(uri, uri=True)
        ) as conn:

            names = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' "
                    "ORDER BY name"
                )
            ]

            for name in names:

                escaped = name.replace('"', '""')

                columns = [
                    row[1]
                    for row in conn.execute(
                        f'PRAGMA table_info("{escaped}")'
                    )
                ]

                count = conn.execute(
                    f'SELECT COUNT(*) FROM "{escaped}"'
                ).fetchone()[0]

                tables.append({
                    "name": name,
                    "rows": count,
                    "columns": columns,
                })

    except sqlite3.DatabaseError as exc:
        return {
            "status": "error",
            "error": str(exc),
            "tables": [],
        }

    return {
        "status": "ok",
        "tables": tables,
    }


# ============================================
# 6. 只读检查Chroma Collections
# ============================================

def scan_chroma(
    folder: Path,
    client_factory: Callable[[Path], Any] | None = None,
) -> dict:

    if not folder.is_dir():
        return {
            "status": "missing",
            "collections": [],
            "names": [],
        }

    if client_factory is None:

        try:
            import chromadb

        except ImportError:
            return {
                "status": "dependency_missing",
                "collections": [],
                "names": [],
            }

        client_factory = (
            lambda path: chromadb.PersistentClient(
                path=str(path)
            )
        )

    try:
        client = client_factory(folder)

        # 兼容返回Collection对象或字符串的版本
        listed = client.list_collections()

        collection_names = [
            item if isinstance(item, str) else item.name
            for item in listed
        ]

        collections = []
        all_sources = set()
        metadata_keys = set()

        for name in collection_names:

            collection = client.get_collection(
                name=name
            )

            total = collection.count()

            without_source = 0

            # 每次只读500条元数据
            for offset in range(0, total, 500):

                batch = collection.get(
                    limit=min(500, total - offset),
                    offset=offset,
                    include=["metadatas"],
                )

                for metadata in (
                    batch.get("metadatas") or []
                ):

                    if isinstance(metadata, dict):
                        metadata_keys.update(
                            metadata.keys()
                        )

                    filename = filename_from_record(
                        {"metadata": metadata}
                    )

                    if filename:
                        all_sources.add(filename)
                    else:
                        without_source += 1

            collections.append({
                "name": name,
                "vectors": total,
                "without_source": without_source,
            })

        return {
            "status": "ok",
            "collections": collections,
            "names": sorted(all_sources),
            "metadata_keys": sorted(metadata_keys),
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "collections": [],
            "names": [],
        }


# ============================================
# 7. 汇总审计结果
# ============================================

def audit_knowledge(
    root: Path,
    client_factory=None,
) -> dict:

    root = Path(root)

    books = scan_books(root / "books")

    cleaned = scan_cleaned(
        root / "data" / "books_clean.json"
    )

    sqlite = scan_sqlite(
        root / "data" / "knowledge.db"
    )

    chroma = scan_chroma(
        root / "data" / "chroma",
        client_factory,
    )

    book_names = {
        name.casefold()
        for name in books["names"]
    }

    cleaned_names = {
        name.casefold()
        for name in cleaned["names"]
    }

    chroma_names = {
        name.casefold()
        for name in chroma["names"]
    }

    warnings = []

    if books["duplicate_names"]:
        warnings.append(
            "原始教材存在同名文件，"
            "仅按文件名比较可能误判。"
        )

    if cleaned["status"] != "ok":
        warnings.append(
            "无法可靠确认清洗数据的来源映射。"
        )

    elif cleaned["without_source"]:
        warnings.append(
            "部分清洗记录没有可识别的来源文件名。"
        )

    if chroma["status"] != "ok":
        warnings.append(
            "目前无法确认Chroma真实索引覆盖。"
        )

    else:
        unknown = sum(
            item["without_source"]
            for item in chroma["collections"]
        )

        if unknown:
            warnings.append(
                "部分Chroma向量没有可识别的教材来源。"
            )

    # 无法识别来源时不计算所谓覆盖率
    missing_cleaned = (
        sorted(book_names - cleaned_names)
        if cleaned["status"] == "ok"
        else None
    )

    missing_indexed = (
        sorted(cleaned_names - chroma_names)
        if (
            cleaned["status"] == "ok"
            and chroma["status"] == "ok"
            and chroma_names
        )
        else None
    )

    return {
        "books": books,
        "cleaned": cleaned,
        "sqlite": sqlite,
        "chroma": chroma,

        "comparison": {
            "books_not_in_cleaned": missing_cleaned,
            "cleaned_not_in_chroma": missing_indexed,
        },

        "warnings": warnings,
    }
