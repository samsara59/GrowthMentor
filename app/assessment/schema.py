
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BehaviorEvidence(BaseModel):
    """从用户经历中提取的行为证据。"""

    model_config = ConfigDict(extra="forbid")

    # 对行为的简短描述
    description: str = Field(min_length=2)

    # 原始证据来自用户行动还是结果
    source_field: Literal["action", "outcome"]

    # 必须是用户原话中的连续片段
    quote: str = Field(min_length=1)


class StrengthHypothesis(BaseModel):
    """需要进一步验证的候选优势方向。"""

    model_config = ConfigDict(extra="forbid")

    # 候选方向名称
    name: str = Field(min_length=2)

    # 为什么提出这个方向
    interpretation: str = Field(min_length=2)

    # 关联哪些行为观察，列表索引从0开始
    evidence_indices: list[int] = Field(min_length=1)


class AssessmentDraft(BaseModel):
    """一次经历的初步分析结果。"""

    model_config = ConfigDict(extra="forbid")

    observations: list[BehaviorEvidence] = Field(
        default_factory=list,
        max_length=3,
    )

    hypotheses: list[StrengthHypothesis] = Field(
        default_factory=list,
        max_length=2,
    )

    # 用来验证或反驳候选方向的下一轮问题
    follow_up_question: str = Field(min_length=5)

    # 当前分析的局限性
    limitation: str = Field(min_length=5)
