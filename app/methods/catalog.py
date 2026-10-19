
"""导师方法目录：理论可以替换，来源需要核验。"""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================
# 1. 定义导师方法
# ============================================

class MethodSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method_id: str = Field(min_length=1)

    name: str = Field(min_length=1)

    kind: Literal[
        "author_method",
        "course",
        "philosophical_lens",
        "historical_lens",
    ]

    intended_use: str = Field(min_length=1)

    # 用于以后检索来源的名称线索
    source_hints: list[str] = Field(
        default_factory=list
    )

    status: Literal[
        "catalog_only",
        "source_verified",
    ] = "catalog_only"

    # 真实核验后填写来源ID
    verified_source_ids: list[str] = Field(
        default_factory=list
    )

    caution: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_verified_sources(self):

        if (
            self.status == "source_verified"
            and not self.verified_source_ids
        ):
            raise ValueError(
                "启用为来源已核验方法前，"
                "必须提供已核验来源ID"
            )

        return self


# ============================================
# 2. 导师方法注册表
# ============================================

class MethodCatalog:

    def __init__(self):
        self._items: dict[str, MethodSpec] = {}

    def register(self, method: MethodSpec) -> None:

        if method.method_id in self._items:
            raise ValueError(
                f"重复的方法：{method.method_id}"
            )

        self._items[method.method_id] = method

    def get(self, method_id: str) -> MethodSpec:
        return self._items[method_id]

    def all(self) -> list[MethodSpec]:
        return list(self._items.values())


# ============================================
# 3. 当前方法目录
# ============================================

def build_method_catalog() -> MethodCatalog:

    catalog = MethodCatalog()

    catalog.register(
        MethodSpec(
            method_id="yagi_self_understanding",

            name="八木仁平：自我理解方法",

            kind="author_method",

            intended_use=(
                "组织喜欢、擅长、重视之事的"
                "探索与实际验证"
            ),

            source_hints=[
                "世界一やさしい「やりたいこと」の見つけ方"
            ],

            caution=(
                "具体提问与定义必须先核实"
                "真实文本来源"
            ),
        )
    )

    catalog.register(
        MethodSpec(
            method_id="local_strengths_courses",

            name="本地优势课程方法包",

            kind="course",

            intended_use=(
                "为具体能力表现寻找理论描述、"
                "行为证据及适用边界"
            ),

            source_hints=[
                "理论课14：分析力发挥.docx",
                "【29行动力解析课】-音频.pdf",
            ],

            caution=(
                "文件曾被解析，不等于Chroma已收录，"
                "也不等于理论已核验"
            ),
        )
    )

    catalog.register(
        MethodSpec(
            method_id="contradiction_lens",

            name="《矛盾论》分析视角",

            kind="philosophical_lens",

            intended_use=(
                "在具体条件下分析相互制约的"
                "问题与主要问题"
            ),

            source_hints=[
                "矛盾论"
            ],

            caution=(
                "仅为问题分析方法；"
                "需要核对原文，不替代用户决策"
            ),
        )
    )

    catalog.register(
        MethodSpec(
            method_id="practice_feedback_lens",

            name="《实践论》实践反馈视角",

            kind="philosophical_lens",

            intended_use=(
                "将认识、行动、结果及修正"
                "连接为反馈循环"
            ),

            source_hints=[
                "实践论"
            ],

            caution=(
                "应用解释需注明来源及适用条件"
            ),
        )
    )

    catalog.register(
        MethodSpec(
            method_id="historical_trajectory_lens",

            name="历史人物成长轨迹比较",

            kind="historical_lens",

            intended_use=(
                "比较所处阶段、条件、行动、"
                "反馈与可迁移策略"
            ),

            source_hints=[],

            caution=(
                "只有实际核验史料后才能生成案例；"
                "不模拟历史人物真实发言"
            ),
        )
    )

    return catalog
