
from dataclasses import dataclass
from typing import Literal

from app.ingestion.schema import DocumentRecord
from app.knowledge.schema import KnowledgeItem
from app.knowledge.extractors import KnowledgeExtractor
from app.knowledge.evidence_validator import validate_knowledge
from app.knowledge.semantic_validator import SemanticValidator


ReviewStatus = Literal[
    "supported",
    "unsupported",
    "uncertain",
    "evidence_invalid",
    "review_error",
]


@dataclass(frozen=True)
class ReviewResult:
    """一条知识的自动审核结果。"""

    item: KnowledgeItem
    status: ReviewStatus
    reason: str


def review_record(
    record: DocumentRecord,
    extractor: KnowledgeExtractor,
    validator: SemanticValidator,
) -> list[ReviewResult]:
    """提取知识，检查证据并执行语义审核。"""

    # 1. 提取候选知识
    items = extractor.extract(record)

    results = []

    for item in items:

        # 2. 检查原文引用
        if not validate_knowledge(item, [record]):
            results.append(
                ReviewResult(
                    item=item,
                    status="evidence_invalid",
                    reason="原文引用无法定位",
                )
            )
            continue

        # 3. 调用语义核验器
        try:
            judgment = validator.validate(
                item,
                [record],
            )

        except (ValueError, RuntimeError) as exc:
            results.append(
                ReviewResult(
                    item=item,
                    status="review_error",
                    reason=str(exc),
                )
            )
            continue

        # 4. 保留判断和理由
        results.append(
            ReviewResult(
                item=item,
                status=judgment.verdict,
                reason=judgment.reason,
            )
        )

    return results
