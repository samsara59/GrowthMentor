
import argparse
import json
from pathlib import Path
from dataclasses import replace
from app.ingestion.loader import load_document
from app.ingestion.file_detector import detect_file_format
from app.ingestion.cleaner import clean_records
ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = ROOT / "books"
DATA_DIR = ROOT / "data"

SUPPORTED = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}


def check_quality(records):
    """初步标记可能存在质量问题的教材。"""

    if not records:
        return ["未提取到正文"]

    text = "\n".join(record.text for record in records)
    lines = [line.strip() for line in text.splitlines()
             if line.strip()]

    # 暂时去掉明显的网址行，估计正文长度
    content_lines = [
        line for line in lines
        if not line.startswith(("http://", "https://"))
    ]

    flags = []

    if len("".join(content_lines)) < 150:
        flags.append("正文可能不足")

    noise_count = sum(
        line.startswith(("http://", "https://"))
        or "返回前一页" in line
        or "包含该课程的还有" in line
        for line in lines
    )

    if lines and noise_count / len(lines) > 0.25:
        flags.append("网页导航信息较多")

    return flags


def ingest_directory(
    folder,
    limit=None,
    ocr_fallback=False,
    filename=None,
    source_map=None
):
    """批量读取教材，单个文件失败不影响其他文件。"""

    if not folder.is_dir():
        raise FileNotFoundError(f"教材目录不存在: {folder}")

    files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED
    )

# 指定文件时，只匹配 books 文件夹内的文件名
    if filename is not None:
        files = [
            path for path in files
            if path.name == filename
        ]

        if not files:
            raise FileNotFoundError(
                f"未找到支持的教材文件: {filename}"
            )

    elif limit is not None:
        files = files[:limit]

    all_records = []
    report = []

    if source_map is None:
        source_map = {}
    
    for path in files:
        print(f"\n正在处理：{path.name}")

        mapping = source_map.get(path.name)
        read_path = path

        try:
            # 1. 如果存在映射，选择修复或转换副本
            if mapping is not None:
                mapped_path = Path(mapping["read_path"])

                if not mapped_path.is_absolute():
                    mapped_path = ROOT / mapped_path

                read_path = mapped_path

                if not read_path.is_file():
                    raise FileNotFoundError(
                        f"映射副本不存在: {read_path}"
                    )

            # 2. 检查实际读取文件的格式
            if read_path.suffix.lower() == ".docx":
                info = detect_file_format(read_path)

                if not info.extension_matches:
                    status = (
                        "needs_conversion"
                        if info.detected_format == "doc"
                        else "format_mismatch"
                    )

                    report.append({
                        "file": path.name,
                        "status": status,
                        "detected_format": info.detected_format,
                        "extension_matches": False,
                        "record_count": 0,
                        "flags": ["文件格式不匹配"]
                    })

                    continue

            # 3. 使用实际文件路径读取内容
            if read_path.suffix.lower() == ".pdf" and ocr_fallback:
                records = load_document(
                    read_path,
                    ocr_fallback=True
                )
            else:
                records = load_document(read_path)

            # 4. 恢复原始教材名称
            records = [
                replace(record, source=path.name)
                for record in records
            ]

            flags = check_quality(records)

            # 5. 保留来源映射中的质量警告
            source_warnings = (
                mapping.get("warnings", [])
                if mapping is not None
                else []
            )

            all_records.extend(
                record.to_dict() for record in records
            )

            # 6. 生成质量报告
            result = {
                "file": path.name,
                "status": (
                    "needs_review"
                    if flags or source_warnings
                    else "prelim_ok"
                ),
                "record_count": len(records),
                "flags": flags
            }

            if mapping is not None:
                result["provenance"] = {
                    "read_path": mapping["read_path"],
                    "operation": mapping["operation"],
                    "warnings": source_warnings
                }

            report.append(result)

            print(f"提取记录：{len(records)}")
            print(f"质量提示：{flags or '暂无明显异常'}")

            if source_warnings:
                print(f"来源警告：{source_warnings}")

        except Exception as exc:
            print(f"读取失败：{exc}")

            error_result = {
                "file": path.name,
                "status": "error",
                "record_count": 0,
                "error": str(exc)
            }

            if mapping is not None:
                error_result["provenance"] = mapping

            report.append(error_result)

    return all_records, report



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--ocr",
        action="store_true",
        help="启用扫描 PDF 的 OCR 回退"
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="指定 books 文件夹中的教材文件名"
    )

    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于等于 1")
    if args.file is not None and args.limit is not None:
        parser.error("--file 和 --limit 不能同时使用")

    source_map_path = DATA_DIR / "source_map.json"

    if source_map_path.is_file():
        source_map = json.loads(
            source_map_path.read_text(encoding="utf-8")
        )
    else:
        source_map = {}

    records, report = ingest_directory(
        BOOK_DIR,
        limit=args.limit,
        ocr_fallback=args.ocr,
        filename=args.file,
        source_map=source_map
    )
    cleaned_records, cleaning_report = clean_records(
        records
    )

    print(f"清洗前记录数：{len(records)}")
    print(f"清洗后记录数：{len(cleaned_records)}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 小规模测试与完整导入分别保存，避免覆盖
    if args.limit is None:
        raw_name = "books_raw.json"
        report_name = "ingestion_report.json"
    else:
        raw_name = "books_sample.json"
        report_name = "ingestion_sample_report.json"
    if args.limit is None:
        clean_name = "books_clean.json"
        cleaning_report_name = "cleaning_report.json"
    else:
        clean_name = "books_sample_clean.json"
        cleaning_report_name = "cleaning_sample_report.json"
    outputs = [
        (raw_name, records),
        (report_name, report),
        (clean_name, cleaned_records),
        (cleaning_report_name, cleaning_report),
    ]
    if args.file is not None:
        file_path = Path(args.file)

        file_id = (
            f"{file_path.stem}_"
            f"{file_path.suffix.lstrip('.').lower()}"
        )

        run_type = "ocr" if args.ocr else "text"

        output_dir = (
            DATA_DIR / "file_checks" / run_type / file_id
        )

    elif args.ocr:
        output_dir = DATA_DIR / (
            "ocr_sample"
            if args.limit is not None
            else "ocr_full"
        )

    else:
        output_dir = DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in outputs:
        output_path = output_dir / filename

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                content, f,
                ensure_ascii=False,
                indent=2
            )

        print(f"已保存：{output_path}")

    print(f"\n共提取 {len(records)} 条记录")
    print(f"共检查 {len(report)} 份教材")


if __name__ == "__main__":
    main()
