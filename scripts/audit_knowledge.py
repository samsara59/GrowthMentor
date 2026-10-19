
"""检查本地优势教材的知识库覆盖情况。"""

import json
from pathlib import Path

from app.knowledge.coverage_audit import (
    audit_knowledge,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:

    print("=" * 55)
    print("GrowthMentor-Agent 本地教材知识库审计")
    print("=" * 55)

    # ========================================
    # 1. 执行只读审计
    # ========================================

    report = audit_knowledge(ROOT)

    books = report["books"]
    cleaned = report["cleaned"]
    sqlite = report["sqlite"]
    chroma = report["chroma"]

    # ========================================
    # 2. 原始文件统计
    # ========================================

    print("\n【1. 原始教材】")

    print("状态：", books["status"])
    print("文件数量：", books["file_count"])

    # ========================================
    # 3. 清洗文本统计
    # ========================================

    print("\n【2. 清洗文本】")

    print("状态：", cleaned["status"])

    print(
        "文本记录数量：",
        cleaned.get("record_count"),
    )

    print(
        "可识别的来源文件数量：",
        len(cleaned["names"]),
    )

    print(
        "没有来源的记录：",
        cleaned.get("without_source"),
    )

    print(
        "记录字段示例：",
        cleaned.get("sample_record_keys", []),
    )

    # ========================================
    # 4. SQLite统计
    # ========================================

    print("\n【3. knowledge.db】")

    print("状态：", sqlite["status"])

    for table in sqlite["tables"]:

        print(
            f"\n表名：{table['name']}"
        )

        print(
            "记录数：",
            table["rows"],
        )

        print(
            "字段：",
            table["columns"],
        )

    # ========================================
    # 5. Chroma统计
    # ========================================

    print("\n【4. Chroma向量索引】")

    print("状态：", chroma["status"])

    print(
        "识别的来源文件数量：",
        len(chroma["names"]),
    )

    print(
        "元数据字段：",
        chroma.get("metadata_keys", []),
    )

    for collection in chroma["collections"]:

        print(
            "\nCollection：",
            collection["name"],
        )

        print(
            "向量数量：",
            collection["vectors"],
        )

        print(
            "无可识别来源数量：",
            collection["without_source"],
        )

    # ========================================
    # 6. 比较文件覆盖
    # ========================================

    print("\n【5. 覆盖检查】")

    comparisons = report["comparison"]

    for key in (
        "books_not_in_cleaned",
        "cleaned_not_in_chroma",
    ):

        missing = comparisons[key]

        if missing is None:

            print(
                key,
                "：无法可靠计算",
            )

        else:

            print(
                key,
                "：",
                len(missing),
            )

            for name in missing[:10]:
                print("  -", name)

    # ========================================
    # 7. 检查提示
    # ========================================

    print("\n【6. 检查提示】")

    for warning in report["warnings"]:
        print("-", warning)

    # ========================================
    # 8. 保存本地审计报告
    # ========================================

    output_path = (
        ROOT / "data" / "knowledge_coverage_audit.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n审计完成。")

    print(
        "完整报告保存在本机：",
        output_path,
    )


if __name__ == "__main__":
    main()
