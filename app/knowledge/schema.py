
from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """一条知识对应的教材证据。"""

    source: str = Field(min_length=1)
    location: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class KnowledgeItem(BaseModel):
    """统一的知识提取结果。"""

    concept: str = Field(min_length=1)

    statement: str = Field(min_length=1)

    knowledge_type: Literal[
        "definition",
        "principle",
        "example",
        "exercise"
    ]

    evidence: list[Evidence] = Field(min_length=1)

    extraction_method: Literal[
        "rule",
        "local_llm",
        "cloud_llm"
    ]
