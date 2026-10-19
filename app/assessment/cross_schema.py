
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CrossEvidence(BaseModel):
    """一条来自指定历史经历的原文证据。"""

    model_config = ConfigDict(extra="forbid")

    # 经历编号，从1开始，对应/history显示的编号
    experience_number: int = Field(ge=1)

    # 原文所在的位置
    source_field: Literal["action", "outcome"]

    # 用户原话中的连续片段
    quote: str = Field(min_length=2)


class CrossCandidate(BaseModel):
    """跨经历重复出现的候选优势方向。"""

    model_config = ConfigDict(extra="forbid")

    # 候选方向名称，不是最终优势标签
    name: str = Field(min_length=2)

    # 根据多条经历提出的解释
    interpretation: str = Field(min_length=5)

    # 必须关联至少两条不同经历
    supports: list[CrossEvidence] = Field(
        min_length=2,
        max_length=6,
    )

    # 可能削弱该假设的证据线索
    possible_counterevidence: list[CrossEvidence] = Field(
        default_factory=list,
        max_length=3,
    )

    # 这项候选方向还有哪些不确定性
    uncertainty: str = Field(min_length=5)


class CrossAssessmentDraft(BaseModel):
    """多条经历的综合分析报告。"""

    model_config = ConfigDict(extra="forbid")

    # 可以没有符合条件的重复模式
    candidates: list[CrossCandidate] = Field(
        default_factory=list,
        max_length=3,
    )

    # 下一轮最值得提出的问题
    next_question: str = Field(min_length=5)

    # 综合分析的总体局限
    limitations: str = Field(min_length=5)
