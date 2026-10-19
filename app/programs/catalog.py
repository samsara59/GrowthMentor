
"""成长任务目录：定义做什么。"""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# ============================================
# 1. 成长任务的描述
# ============================================

class ProgramSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    program_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    objective: str = Field(min_length=1)

    phases: list[str] = Field(min_length=1)

    suggested_method_ids: list[str] = Field(
        default_factory=list
    )

    status: Literal[
        "planned",
        "legacy_prototype",
        "active",
    ] = "planned"


# ============================================
# 2. 成长任务注册表
# ============================================

class ProgramCatalog:

    def __init__(self):
        self._items: dict[str, ProgramSpec] = {}

    def register(
        self,
        program: ProgramSpec,
    ) -> None:

        if program.program_id in self._items:
            raise ValueError(
                f"重复的成长任务：{program.program_id}"
            )

        self._items[program.program_id] = program

    def get(
        self,
        program_id: str,
    ) -> ProgramSpec:

        return self._items[program_id]

    def all(self) -> list[ProgramSpec]:
        return list(self._items.values())


# ============================================
# 3. 当前的成长任务目录
# ============================================

def build_program_catalog() -> ProgramCatalog:

    catalog = ProgramCatalog()

    # 第一个目标：自我发现
    catalog.register(
        ProgramSpec(
            program_id="self_discovery",

            name=(
                "自我发现：喜欢、擅长、重要的事"
            ),

            objective=(
                "发现值得通过实践验证的方向，"
                "而不是给出永久标签"
            ),

            phases=[
                "探索",
                "证据核对",
                "形成方向假设",
                "小实验",
                "反馈",
            ],

            suggested_method_ids=[
                "yagi_self_understanding",
                "local_strengths_courses",
            ],

            status="legacy_prototype",
        )
    )

    # 第二个目标：现实困境分析
    catalog.register(
        ProgramSpec(
            program_id="reflective_decision",

            name="现实困境分析与决策",

            objective=(
                "分析具体条件、制约因素，"
                "并选择可检验的行动"
            ),

            phases=[
                "描述问题",
                "识别制约",
                "比较路径",
                "实践",
                "复盘",
            ],

            suggested_method_ids=[
                "contradiction_lens",
                "practice_feedback_lens",
            ],
        )
    )

    # 第三个目标：历史人物成长路径学习
    catalog.register(
        ProgramSpec(
            program_id="historical_learning",

            name="历史人物成长路径学习",

            objective=(
                "基于可信史料比较成长阶段"
                "与可迁移策略"
            ),

            phases=[
                "明确问题",
                "检索史料",
                "比较条件",
                "提取策略",
                "实践复盘",
            ],

            suggested_method_ids=[
                "historical_trajectory_lens",
            ],
        )
    )

    return catalog
