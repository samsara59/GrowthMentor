
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
from app.knowledge.review_pipeline import review_record
from app.knowledge.store import (
    save_review_results,
    load_review_results,
)


ROOT = Path(__file__).resolve().parents[1]

REVIEW_FILE = (
    ROOT / "data"
    / "knowledge_reviews"
    / "real_sample_review.json"
)

RECORDS_FILE = ROOT / "data" / "books_clean.json"

DB_FILE = ROOT / "data" / "knowledge.db"


class SavedKnowledgeExtractor:
    """复用已提取的知识，不再调用提取模型。"""

    def __init__(self, item: KnowledgeItem):
        self.item = item

    def extract(
        self,
        record: DocumentRecord,
    ) -> list[KnowledgeItem]:
        return [self.item]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="要审核的知识编号，从1开始",
    )

    args = parser.parse_args()

    # 1. 加载已经提取的候选知识
    review_data = json.loads(
        REVIEW_FILE.read_text(encoding="utf-8")
    )

    if not 1 <= args.index <= len(review_data):
        raise ValueError("知识编号超出范围")

    item = KnowledgeItem.model_validate(
        review_data[args.index - 1]["knowledge"]
    )

    # 2. 确认本次知识引用的是同一条原文记录
    locations = {
        (e.source, e.location)
        for e in item.evidence
    }

    if len(locations) != 1:
        raise ValueError(
            "当前脚本只支持引用同一条教材记录的知识"
        )

    source, location = next(iter(locations))

    # 3. 加载对应教材记录
    raw_records = json.loads(
        RECORDS_FILE.read_text(encoding="utf-8")
    )

    matches = [
        record
        for record in raw_records
        if record.get("source") == source
        and record.get("location") == location
    ]

    if len(matches) != 1:
        raise ValueError(
            f"预期找到1条原文记录，实际找到{len(matches)}条"
        )

    record = DocumentRecord(**matches[0])

    # 4. 创建语义核验模型
    client = OllamaClient(
        model="gemma3:4b",
        num_gpu=0,
        timeout=300,
        response_format=(
            SemanticJudgment.model_json_schema()
        ),
    )

    validator = SemanticValidator(client)

    # 5. 调用统一审核流水线
    results = review_record(
        record=record,
        extractor=SavedKnowledgeExtractor(item),
        validator=validator,
    )

    result = results[0]

    print("\n=== 审核结果 ===")
    print("概念：", result.item.concept)
    print("状态：", result.status)
    print("理由：", result.reason)

    # 6. 将审核结果持久保存
    save_review_results(
        DB_FILE,
        results,
    )

    print("\n已保存数据库：", DB_FILE)

    # 7. 从数据库重新读取
    saved_results = load_review_results(DB_FILE)

    print("\n=== 数据库验证 ===")
    print("数据库知识总数：", len(saved_results))

    matching = [
        r for r in saved_results
        if r.item == result.item
    ]

    if len(matching) != 1:
        raise RuntimeError(
            "数据库没有正确保存当前知识"
        )

    print("已恢复概念：", matching[0].item.concept)
    print("已恢复状态：", matching[0].status)
    print("原文来源：", matching[0].item.evidence[0].source)
    print("原文位置：", matching[0].item.evidence[0].location)


if __name__ == "__main__":
    main()
