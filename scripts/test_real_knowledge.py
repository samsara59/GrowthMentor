
import json
from pathlib import Path

from app.ingestion.schema import DocumentRecord
from app.knowledge.extractors import (
    RuleExtractor,
    run_extractor,
)
from app.knowledge.llm_extractor import (
    LLMExtractor,
    KnowledgeBatch,
)
from app.knowledge.ollama_client import OllamaClient


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "books_clean.json"


def select_record(records):
    """选择一条适合首次测试的真实教材记录。"""

    candidates = []

    for item in records:
        text = item.get("text", "")

        if not isinstance(text, str):
            continue

        if not 150 <= len(text) <= 700:
            continue

        if not all(
            key in item
            for key in ("source", "location", "method")
        ):
            continue

        candidates.append(item)

    if not candidates:
        raise ValueError("没有找到长度合适的教材记录")

    # 优先选择包含明确概念定义的记录
    for item in candidates:
        if "是指" in item["text"]:
            return item

    return candidates[0]


def show_results(name, accepted, rejected):
    """打印提取与证据核验结果。"""

    print(f"\n=== {name} ===")
    print("核验通过：", len(accepted))
    print("核验拒绝：", len(rejected))

    for item in accepted:
        print("\n概念：", item.concept)
        print("类别：", item.knowledge_type)
        print("陈述：", item.statement)

        for evidence in item.evidence:
            print("原文引用：", evidence.quote)

    for item in rejected:
        print("\n被拒绝的概念：", item.concept)


def main():
    # 1. 加载已经清洗的教材数据
    if not DATA_FILE.is_file():
        raise FileNotFoundError(DATA_FILE)

    records = json.loads(
        DATA_FILE.read_text(encoding="utf-8")
    )

    # 2. 选择测试记录
    item = select_record(records)

    record = DocumentRecord(
        source=item["source"],
        location=item["location"],
        text=item["text"],
        method=item["method"],
    )

    print("=== 测试教材 ===")
    print("来源：", record.source)
    print("位置：", record.location)
    print("文本长度：", len(record.text))

    # 3. 传统规则提取
    rule_accepted, rule_rejected = run_extractor(
        RuleExtractor(),
        record,
    )

    show_results(
        "传统规则提取",
        rule_accepted,
        rule_rejected,
    )

    # 4. 本地模型提取
    client = OllamaClient(
        model="gemma3:4b",
        num_gpu=0,
        timeout=300,
        response_format=KnowledgeBatch.model_json_schema(),
    )

    llm_extractor = LLMExtractor(
        client=client,
        method="local_llm",
    )

    llm_accepted, llm_rejected = run_extractor(
        llm_extractor,
        record,
    )

    show_results(
        "Gemma 3 提取",
        llm_accepted,
        llm_rejected,
    )


    # 保存真实教材知识的人工审核任务
    review_items = []

    for index, item in enumerate(llm_accepted, start=1):
        review_items.append({
            "id": index,
            "knowledge": item.model_dump(),
            "evidence_verified": True,
            "semantic_support": "pending",
            "category_correct": "pending",
            "duplicate": "pending",
            "review_note": ""
        })

    output_dir = ROOT / "data" / "knowledge_reviews"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "real_sample_review.json"

    if output_path.exists():
        print("\n审核文件已存在，未覆盖：", output_path)
    else:
        output_path.write_text(
            json.dumps(
                review_items,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print("\n已生成知识审核文件：", output_path)

if __name__ == "__main__":
    main()
