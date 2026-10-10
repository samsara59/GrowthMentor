
import re


# 明确属于页面操作的独立文本
NOISE_LINES = {
    "返回顶部",
    "返回首页",
    "上一页",
    "下一页",
    "点击展开全文",
    "收起全文",
}


def clean_text(text: str) -> tuple[str, list[str]]:
    """清洗一段文字，返回清洗结果及删除记录。"""

    # 1. 统一换行符
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 2. 删除不可见字符
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    kept_lines = []
    removed_lines = []

    # 3. 逐行过滤
    for raw_line in text.split("\n"):

        line = re.sub(
            r"[ \t\u00a0]+",
            " ",
            raw_line
        ).strip()

        if not line:
            continue

        if line in NOISE_LINES:
            removed_lines.append(line)
            continue

        kept_lines.append(line)

    # 4. 重新组成正文
    cleaned_text = "\n".join(kept_lines)

    return cleaned_text, removed_lines


def clean_records(
    records: list[dict]
) -> tuple[list[dict], list[dict]]:
    """统一清洗所有教材记录，不修改原始输入。"""

    cleaned_records = []
    cleaning_report = []

    for record in records:

        raw_text = record["text"]

        cleaned_text, removed = clean_text(raw_text)

        # 对需要人工判断的内容添加标记
        flags = []

        if re.search(
            r"(?m)^https?://\S+\s*$",
            cleaned_text
        ):
            flags.append("包含独立网址行")

        if not cleaned_text:
            flags.append("清洗后没有正文")
        else:
            # 创建新字典，保留原始记录
            cleaned_record = {
                **record,
                "text": cleaned_text
            }

            cleaned_records.append(cleaned_record)

        # 记录本次清洗过程
        cleaning_report.append({
            "source": record["source"],
            "location": record["location"],
            "raw_length": len(raw_text),
            "cleaned_length": len(cleaned_text),
            "removed_lines": removed,
            "flags": flags
        })

    return cleaned_records, cleaning_report
