
"""只读教材和索引，生成来源覆盖与原文核验报告。"""

import argparse
import json

from collections import Counter
from datetime import datetime
from pathlib import Path

from app.knowledge.coverage_audit import (
    filename_from_record,
)

from app.knowledge.source_provenance import (
    SourceCatalog,
    suspected_ocr,
    verify_excerpt,
)

from scripts.build_source_index import (
    load_records,
    extract_text,
)


ROOT = Path(__file__).resolve().parents[1]


# ============================================
# 1. 汇总教材、清洗记录和Chroma
# ============================================

def run(
    root: Path,
    samples: int = 12,
) -> dict:

    catalog = SourceCatalog(
        root / "books"
    )

    clean_path = (
        root / "data" / "books_clean.json"
    )

    if not clean_path.is_file():

        raise FileNotFoundError(
            f"缺少清洗数据：{clean_path}"
        )

    records = load_records(
        clean_path
    )

    clean_files = set()
    clean_status = Counter()

    unresolved_examples = []

    # ========================================
    # 2. 核对每条清洗记录的来源
    # ========================================

    for record in records:

        if not isinstance(record, dict):

            clean_status[
                "non_object_record"
            ] += 1

            continue

        if not extract_text(record):

            clean_status[
                "missing_text"
            ] += 1

        source = filename_from_record(
            record
        )

        status, file_path = (
            catalog.resolve(source)
        )

        clean_status[status] += 1

        if file_path:

            clean_files.add(
                catalog.display(file_path)
            )

        elif len(unresolved_examples) < 15:

            unresolved_examples.append({
                "source": str(source)[:200],
                "reason": status,
            })

    # ========================================
    # 3. 核对真正存在的Chroma集合
    # ========================================

    chroma_status = "not_found"
    index_count = 0

    index_source_files = set()
    index_counter = Counter()

    sample_results = []

    chroma_folder = (
        root / "data" / "chroma"
    )

    if chroma_folder.is_dir():

        try:

            import chromadb

            client = (
                chromadb.PersistentClient(
                    path=str(chroma_folder)
                )
            )

            collection = client.get_collection(
                name="growthmentor_source_chunks_v1"
            )

            index_count = collection.count()

            chroma_status = "ok"

            # =================================
            # 4. 分批核对全部向量的来源
            # =================================

            for offset in range(
                0,
                index_count,
                500,
            ):

                batch = collection.get(
                    offset=offset,
                    limit=500,
                    include=["metadatas"],
                )

                ids = batch.get(
                    "ids"
                ) or []

                if not ids:
                    break

                for meta in (
                    batch.get("metadatas") or []
                ):

                    source = (
                        meta.get("source")
                        if isinstance(meta, dict)
                        else None
                    )

                    status, file_path = (
                        catalog.resolve(source)
                    )

                    index_counter[
                        status
                    ] += 1

                    if file_path:

                        index_source_files.add(
                            catalog.display(
                                file_path
                            )
                        )

            # =================================
            # 5. 从整个集合分层选取样本
            # =================================

            sample_count = min(
                samples,
                index_count,
            )

            for k in range(
                sample_count
            ):

                index = min(
                    index_count - 1,

                    int(
                        (k + 0.5)
                        * index_count
                        / sample_count
                    ),
                )

                hit = collection.get(
                    offset=index,
                    limit=1,
                    include=[
                        "documents",
                        "metadatas",
                    ],
                )

                ids = hit.get(
                    "ids"
                ) or []

                if not ids:
                    continue

                meta = (
                    hit.get("metadatas")
                    or [None]
                )[0] or {}

                doc = (
                    hit.get("documents")
                    or [""]
                )[0] or ""

                outcome = verify_excerpt(
                    catalog,
                    meta.get("source", ""),
                    doc,
                )

                sample_results.append({
                    "vector_id": ids[0],
                    **outcome.as_dict(),
                })

        except Exception as exc:

            chroma_status = (
                f"error:{type(exc).__name__}:"
                f"{str(exc)[:160]}"
            )

    # ========================================
    # 6. 找出未从清洗记录映射的文件
    # ========================================

    never_in_clean = [
        p
        for p in catalog.paths

        if catalog.display(p)
        not in clean_files
    ]

    # ========================================
    # 7. 生成本地JSON报告
    # ========================================

    report = {
        "generated_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),

        "disclaimer": (
            "原文字符匹配不等于语义结论"
            "得到教材支持；疑似扫描仅抽样；"
            "本程序不执行OCR或重建索引。"
        ),

        "original_file_count": (
            len(catalog.paths)
        ),

        "clean_record_count": (
            len(records)
        ),

        "clean_unique_original_sources": (
            len(clean_files)
        ),

        "clean_record_resolution": (
            dict(clean_status)
        ),

        "clean_unresolved_examples": (
            unresolved_examples
        ),

        "original_files_not_mapped_from_clean": [
            catalog.display(p)
            for p in never_in_clean
        ],

        "source_index_status": (
            chroma_status
        ),

        "source_index_vector_count": (
            index_count
        ),

        "source_index_unique_original_sources": (
            len(index_source_files)
        ),

        "source_index_resolution": (
            dict(index_counter)
        ),

        "original_files_not_mapped_from_index": [
            catalog.display(p)
            for p in catalog.paths
            if catalog.display(p)
            not in index_source_files
        ],

        "sample_verification": (
            sample_results
        ),

        "ocr_or_extraction_candidates": (
            suspected_ocr(
                catalog,
                never_in_clean,
            )
        ),
    }

    return report


# ============================================
# 8. Windows命令行入口
# ============================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    if not 0 <= args.samples <= 100:

        parser.error(
            "--samples必须在0到100之间"
        )

    report = run(
        ROOT,
        samples=args.samples,
    )

    path = (
        ROOT / "data"
        / "source_provenance_report.json"
    )

    path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "原始教材数：",
        report["original_file_count"],
    )

    print(
        "清洗记录数：",
        report["clean_record_count"],
    )

    print(
        "唯一映射的清洗来源文件数：",
        report["clean_unique_original_sources"],
    )

    print(
        "未能从清洗记录映射的原始文件数：",
        len(
            report[
                "original_files_not_mapped_from_clean"
            ]
        ),
    )

    print(
        "原文索引状态：",
        report["source_index_status"],
    )

    print(
        "原文索引向量数：",
        report["source_index_vector_count"],
    )

    print(
        "原文索引唯一匹配的文件数：",
        report["source_index_unique_original_sources"],
    )

    print(
        "原文核验抽样状态：",
        dict(
            Counter(
                x["status"]
                for x in report[
                    "sample_verification"
                ]
            )
        ),
    )

    print(
        "OCR或额外提取候选数：",
        len(
            report[
                "ocr_or_extraction_candidates"
            ]
        ),
    )

    print(
        "详细报告：",
        path,
    )

    print(
        "没有执行OCR，"
        "没有修改教材或向量索引。"
    )


if __name__ == "__main__":
    main()
