
"""长期成长导师的稳定数据协议。"""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================
# 1. 用户当前的成长任务
# ============================================

class GrowthTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    program_id: str = Field(min_length=1)

    goal: str = Field(min_length=1)
    context: str = ""

    constraints: list[str] = Field(
        default_factory=list
    )

    status: Literal[
        "open",
        "paused",
        "completed",
    ] = "open"


# ============================================
# 2. 来自用户自述的证据
# ============================================

class UserEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    context: str = ""


# ============================================
# 3. 来自教材或史料的证据
# ============================================

class KnowledgeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)

    source_id: str = Field(min_length=1)
    source_title: str = Field(min_length=1)

    location: str = ""
    quote: str = Field(min_length=1)

    review_status: Literal[
        "retrieved",
        "checked",
    ] = "retrieved"


# ============================================
# 4. 尚待验证的成长假设
# ============================================

class GrowthHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(min_length=1)

    statement: str = Field(min_length=1)

    user_evidence_ids: list[str] = Field(
        default_factory=list
    )

    knowledge_evidence_ids: list[str] = Field(
        default_factory=list
    )

    uncertainty: str = Field(min_length=1)


# ============================================
# 5. 一次成长任务的输出
# ============================================

class GrowthOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    response: str = Field(min_length=1)

    user_evidence: list[UserEvidence] = Field(
        default_factory=list
    )

    knowledge_evidence: list[
        KnowledgeEvidence
    ] = Field(default_factory=list)

    hypotheses: list[GrowthHypothesis] = Field(
        default_factory=list
    )

    next_action: str | None = None

    # ========================================
    # 6. 验证证据之间的引用关系
    # ========================================

    @model_validator(mode="after")
    def check_references(self):

        user_ids = [
            item.evidence_id
            for item in self.user_evidence
        ]

        knowledge_ids = [
            item.evidence_id
            for item in self.knowledge_evidence
        ]

        hypothesis_ids = [
            item.hypothesis_id
            for item in self.hypotheses
        ]

        if (
            len(user_ids) != len(set(user_ids))
            or len(knowledge_ids) != len(set(knowledge_ids))
        ):
            raise ValueError(
                "证据编号必须在各自类别内唯一"
            )

        if len(hypothesis_ids) != len(
            set(hypothesis_ids)
        ):
            raise ValueError(
                "假设编号不能重复"
            )

        for hypothesis in self.hypotheses:

            if not set(
                hypothesis.user_evidence_ids
            ).issubset(user_ids):

                raise ValueError(
                    "假设引用了不存在的用户证据"
                )

            if not set(
                hypothesis.knowledge_evidence_ids
            ).issubset(knowledge_ids):

                raise ValueError(
                    "假设引用了不存在的教材证据"
                )

        return self
