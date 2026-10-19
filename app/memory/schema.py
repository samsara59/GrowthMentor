
from dataclasses import dataclass, field
from typing import Literal

from app.assessment.schema import AssessmentDraft

from app.assessment.cross_schema import (
    CrossAssessmentDraft,
)


Stage = Literal[
    "idle",
    "awaiting_situation",
    "awaiting_action",
    "awaiting_outcome",
    "awaiting_verification",
]


@dataclass(frozen=True)
class ExperienceRecord:
    situation: str
    action: str
    outcome: str


@dataclass(frozen=True)
class VerificationTurn:
    """已经完成分析的一轮验证问答。"""

    question: str
    answer: str

    relation: Literal[
        "supports",
        "challenges",
        "unclear",
    ]

    quote: str
    reason: str


@dataclass
class VerificationSession:
    """某个候选方向的动态验证状态。"""

    candidate_number: int
    candidate_name: str

    # 生成报告时的原始经历指纹
    source_fingerprint: str

    # 当前正在等待用户回答的问题
    pending_question: str

    # 已完成的问答及证据判断
    turns: list[VerificationTurn] = field(
        default_factory=list
    )

    # 模型分析失败时，暂存用户刚才的回答
    pending_answer: str | None = None

    completed: bool = False


@dataclass
class AgentSession:
    """用户完整会话状态。"""

    stage: Stage = "idle"

    # 尚未完成的三阶段访谈
    draft: dict[str, str] = field(
        default_factory=dict
    )

    # 原始经历及单经历分析
    experiences: list[ExperienceRecord] = field(
        default_factory=list
    )

    assessments: list[
        AssessmentDraft | None
    ] = field(default_factory=list)

    # 跨经历综合报告
    cross_report: CrossAssessmentDraft | None = None

    # 报告对应的原始数据版本
    cross_report_fingerprint: str | None = None

    # 当前或最近一次验证会话
    verification: VerificationSession | None = None
