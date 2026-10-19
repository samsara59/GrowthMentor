
import argparse
import json
from pathlib import Path

from app.ingestion.schema import DocumentRecord
from app.knowledge.schema import KnowledgeItem

from app.knowledge.ollama_client import OllamaClient

from app.knowledge.semantic_validator import (
    SemanticJudgment,
    SemanticValidator,
)


ROOT = Path(__file__).resolve().parents[1]

REVIEW_FILE = (
    ROOT / "data" / "knowledge_reviews"
    / "real_sample_review.json"
)

RECORDS_FILE = ROOT / "data" / "books_clean.json"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="审核第几条知识，从1开始",
    )

    args = parser.parse_args()

    # 1. 读取上次保存的候选知识
    review_items = json.loads(
        REVIEW_FILE.read_text(encoding="utf-8")
    )

    if not 1 <= args.index <= len(review_items):
        raise ValueError("知识索引超出范围")

    item = KnowledgeItem.model_validate(
        review_items[args.index - 1]["knowledge"]
    )

    # 2. 根据来源和位置找到原始记录
    all_records = json.loads(
        RECORDS_FILE.read_text(encoding="utf-8")
    )

    source = item.evidence[0].source
    location = item.evidence[0].location

    records = [
        DocumentRecord(**record)
        for record in all_records
        if record.get("source") == source
        and record.get("location") == location
    ]

    if not records:
        raise ValueError("未找到对应的教材记录")

    # 3. 创建语义核验模型
    client = OllamaClient(
        model="gemma3:4b",
        num_gpu=0,
        timeout=300,
        response_format=SemanticJudgment.model_json_schema(),
    )

    validator = SemanticValidator(client)

    # 4. 显示待核验知识
    print("\n=== 待核验知识 ===")
    print("概念：", item.concept)
    print("陈述：", item.statement)
    print("引用数量：", len(item.evidence))

    # 5. 执行原文检查及语义核验
    result = validator.validate(
        item,
        records,
    )

    print("\n=== 语义核验结果 ===")
    print("判断：", result.verdict)
    print("理由：", result.reason)

    # 构造与原文不一致的陈述
    negative_item = item.model_copy(
        update={
            "statement": (
                "Question只指作者提供的答案，"
                "完全不包括学习者提出的问题。"
            )
        }
    )

    print("\n=== 反样本测试 ===")
    print("错误陈述：", negative_item.statement)

    # 使用完全相同的原文证据
    negative_result = validator.validate(
        negative_item,
        records,
    )

    print("判断：", negative_result.verdict)
    print("理由：", negative_result.reason)



if __name__ == "__main__":
    main()
