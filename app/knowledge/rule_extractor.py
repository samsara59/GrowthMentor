
import re

from app.ingestion.schema import DocumentRecord
from app.knowledge.schema import Evidence, KnowledgeItem


# 第一版仅识别行首的明确概念定义
DEFINITION_PATTERN = re.compile(
    r"(?m)^(?P<concept>[\u4e00-\u9fff]{2,12})"
    r"是指"
    r"(?P<definition>[^。！？\n]{4,150})。"
)


def extract_definitions(
    record: DocumentRecord
) -> list[KnowledgeItem]:
    """从一条文档记录提取明确的定义句。"""

    results = []

    for match in DEFINITION_PATTERN.finditer(record.text):

        concept = match.group("concept")
        quote = match.group(0)

        item = KnowledgeItem(
            concept=concept,
            statement=quote,
            knowledge_type="definition",
            evidence=[
                Evidence(
                    source=record.source,
                    location=record.location,
                    quote=quote
                )
            ],
            extraction_method="rule"
        )

        results.append(item)

    return results
