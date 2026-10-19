
from app.ingestion.schema import DocumentRecord
from app.knowledge.schema import Evidence, KnowledgeItem


def validate_evidence(
    evidence: Evidence,
    records: list[DocumentRecord]
) -> bool:
    """检查一条证据是否能够追溯到原文。"""

    # 拒绝只有空格的引用
    if not evidence.quote.strip():
        return False

    for record in records:

        # 1. 文件来源必须匹配
        if record.source != evidence.source:
            continue

        # 2. 页码或段落位置必须匹配
        if record.location != evidence.location:
            continue

        # 3. 引用必须真实出现在原文中
        if evidence.quote in record.text:
            return True

    return False


def validate_knowledge(
    item: KnowledgeItem,
    records: list[DocumentRecord]
) -> bool:
    """一条知识的所有证据都必须通过核验。"""

    return all(
        validate_evidence(evidence, records)
        for evidence in item.evidence
    )
