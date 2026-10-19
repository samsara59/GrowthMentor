
from typing import Protocol

from app.ingestion.schema import DocumentRecord
from app.knowledge.schema import KnowledgeItem
from app.knowledge.rule_extractor import extract_definitions
from app.knowledge.evidence_validator import validate_knowledge


class KnowledgeExtractor(Protocol):
    """所有知识提取器需要遵守的接口。"""

    def extract(
        self,
        record: DocumentRecord
    ) -> list[KnowledgeItem]:
        ...


class RuleExtractor:
    """对现有规则提取函数进行接口封装。"""

    def extract(
        self,
        record: DocumentRecord
    ) -> list[KnowledgeItem]:

        return extract_definitions(record)


def run_extractor(
    extractor: KnowledgeExtractor,
    record: DocumentRecord
) -> tuple[list[KnowledgeItem], list[KnowledgeItem]]:
    """执行知识提取，并区分证据通过与未通过的知识。"""

    items = extractor.extract(record)

    accepted = []
    rejected = []

    for item in items:
        if validate_knowledge(item, [record]):
            accepted.append(item)
        else:
            rejected.append(item)

    return accepted, rejected
