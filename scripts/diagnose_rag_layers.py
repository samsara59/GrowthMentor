
"""诊断知识库覆盖、向量数量及检索重复问题。"""

import hashlib
import json

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

from scripts.try_knowledge_retrieve import build_backend


ROOT = Path(__file__).resolve().parents[1]

AUDIT_PATH = (
    ROOT / "data" / "knowledge_coverage_audit.json"
)


# ============================================
# 1. 将KnowledgeItem转为可比较的数据
# ============================================

def object_to_dict(obj):

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")

    if is_dataclass(obj):
        return asdict(obj)

    if hasattr(obj, "__dict__"):
        return vars(obj)

    raise TypeError(
        f"无法解析对象：{type(obj).__name__}"
    )


# ============================================
# 2. 对知识条目计算内容指纹
# ============================================

def item_fingerprint(item):

    data = object_to_dict(item)

    encoded = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()[:16]


# ============================================
# 3. 输出此前知识库审计结果
# ============================================

def print_audit():

    print("\n========== 知识库覆盖诊断 ==========")

    if not AUDIT_PATH.is_file():

        print(
            "未找到上一步的知识库审计报告。"
        )

        print(
            "请先执行 python -m scripts.audit_knowledge"
        )

        return

    report = json.loads(
        AUDIT_PATH.read_text(
            encoding="utf-8"
        )
    )

    modified = datetime.fromtimestamp(
        AUDIT_PATH.stat().st_mtime
    )

    print(
        "报告生成文件的修改时间：",
        modified.isoformat(
            timespec="seconds"
        ),
    )

    books = report.get("books", {})
    cleaned = report.get("cleaned", {})
    sqlite = report.get("sqlite", {})
    chroma = report.get("chroma", {})

    print("\n[原始教材]")

    print(
        "状态：",
        books.get("status"),
    )

    print(
        "文件数量：",
        books.get("file_count"),
    )

    print("\n[清洗记录]")

    print(
        "状态：",
        cleaned.get("status"),
    )

    print(
        "记录数：",
        cleaned.get("record_count"),
    )

    print(
        "可识别来源文件数：",
        len(cleaned.get("names") or []),
    )

    print("\n[SQLite知识库]")

    print(
        "状态：",
        sqlite.get("status"),
    )

    for table in sqlite.get("tables", []):

        print(
            f"表 {table['name']}："
            f"{table['rows']} 行"
        )

    print("\n[Chroma向量索引]")

    print(
        "状态：",
        chroma.get("status"),
    )

    print(
        "识别的来源文件数：",
        len(chroma.get("names") or []),
    )

    for collection in chroma.get(
        "collections", []
    ):

        print(
            f"Collection {collection['name']}："
            f"{collection['vectors']} 个向量"
        )

        print(
            "  无可识别来源的向量：",
            collection["without_source"],
        )

    print(
        "Chroma元数据字段：",
        chroma.get("metadata_keys", []),
    )

    print("\n[审计提示]")

    for warning in report.get("warnings", []):
        print("-", warning)


# ============================================
# 4. 比较不同查询的知识条目
# ============================================

def check_queries():

    print("\n========== 语义检索诊断 ==========")

    backend = build_backend()

    queries = [
        "分析力",
        "行动力",
        "如何找到真正喜欢的事情",
    ]

    fingerprints = {}

    for query in queries:

        print("\n查询：", query)

        hits = backend.search(
            query=query,
            limit=10,
        )

        print(
            "经过检索器处理后返回：",
            len(hits),
        )

        fingerprints[query] = []

        for number, hit in enumerate(
            hits,
            start=1,
        ):

            review = hit.result
            item = review.item

            # 不输出私有教材正文
            item_data = object_to_dict(item)

            fingerprint = item_fingerprint(item)

            fingerprints[query].append(
                fingerprint
            )

            print(
                f"\n  结果{number}"
            )

            print(
                "  内容指纹：",
                fingerprint,
            )

            print(
                "  概念：",
                str(
                    item_data.get(
                        "concept", ""
                    )
                )[:100],
            )

            print(
                "  知识类型：",
                item_data.get(
                    "knowledge_type"
                ),
            )

            print(
                "  抽取方式：",
                item_data.get(
                    "extraction_method"
                ),
            )

            print(
                "  审核状态：",
                review.status,
            )

            print(
                "  距离：",
                hit.distance,
            )

            evidence = item_data.get(
                "evidence"
            )

            # 只检查证据结构，不打印原文
            if isinstance(evidence, dict):

                print(
                    "  evidence字段：",
                    list(evidence.keys()),
                )

            elif isinstance(evidence, list):

                print(
                    "  evidence类型：list"
                )

                print(
                    "  evidence数量：",
                    len(evidence),
                )

                if (
                    evidence
                    and isinstance(evidence[0], dict)
                ):

                    print(
                        "  首条证据字段：",
                        list(evidence[0].keys()),
                    )

            else:

                print(
                    "  evidence类型：",
                    type(evidence).__name__,
                )

    print("\n========== 查询结果交集 ==========")

    for index, first in enumerate(queries):

        for second in queries[index + 1:]:

            overlap = (
                set(fingerprints[first])
                & set(fingerprints[second])
            )

            print(
                f"{first} ∩ {second}："
                f"{len(overlap)} 个相同内容指纹"
            )

    print(
        "\n说明：内容指纹只检测条目内容是否相同，"
        "不能单独证明来自同一教材文件。"
    )


# ============================================
# 5. 程序入口
# ============================================

def main():

    print_audit()

    check_queries()

    print(
        "\n诊断完成。本程序没有执行索引同步。"
    )


if __name__ == "__main__":
    main()
