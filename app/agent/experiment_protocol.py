
"""第68课修复：由LLM生成A/B实践，保留具体校验错误。"""

import json
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)


# ============================================
# 1. 最终保存的完整实践方案
# ============================================

class ComparisonProtocol(BaseModel):
    """
    对外保持原有数据结构。

    兼容：
    - 第67课AdaptiveMentor
    - 第68课Streamlit
    - PracticeStore中的SQLite JSON
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    option_a: str = Field(
        min_length=2,
        max_length=120,
    )

    option_b: str = Field(
        min_length=2,
        max_length=120,
    )

    question: str = Field(
        min_length=4,
        max_length=200,
    )

    common_problem: str = Field(
        min_length=5,
        max_length=400,
    )

    task_a: str = Field(
        min_length=8,
        max_length=600,
    )

    output_a: str = Field(
        min_length=2,
        max_length=200,
    )

    task_b: str = Field(
        min_length=8,
        max_length=600,
    )

    output_b: str = Field(
        min_length=2,
        max_length=200,
    )

    minutes_per_session: int = Field(
        ge=15,
        le=120,
    )

    constants: list[str] = Field(
        min_length=2,
        max_length=5,
    )

    observations: list[str] = Field(
        min_length=3,
        max_length=6,
    )

    limitations: str = Field(
        min_length=8,
        max_length=300,
    )

    confirmation_question: str = Field(
        min_length=5,
        max_length=200,
    )

    @model_validator(mode="after")
    def check_contrast(self):

        if (
            self.option_a.strip()
            == self.option_b.strip()
        ):
            raise ValueError(
                "两个成长方向必须不同"
            )

        if (
            self.task_a.strip()
            == self.task_b.strip()
        ):
            raise ValueError(
                "A和B的活动内容不能相同"
            )

        for group in (
            self.constants,
            self.observations,
        ):

            if any(
                not isinstance(x, str)
                or not x.strip()
                for x in group
            ):
                raise ValueError(
                    "控制条件和观察项目"
                    "不能包含空文本"
                )

        return self


# ============================================
# 2. 只让LLM生成尚未确定的内容
# ============================================

class _GeneratedContent(BaseModel):
    """
    用户已确定option_a、option_b，
    因此LLM不需要重新生成这两个字段。

    LLM仍然负责具体任务、共同场景、
    观察指标、时间和限制说明。
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    question: str = Field(
        min_length=4,
        max_length=200,
    )

    common_problem: str = Field(
        min_length=5,
        max_length=400,
    )

    task_a: str = Field(
        min_length=8,
        max_length=600,
    )

    output_a: str = Field(
        min_length=2,
        max_length=200,
    )

    task_b: str = Field(
        min_length=8,
        max_length=600,
    )

    output_b: str = Field(
        min_length=2,
        max_length=200,
    )

    minutes_per_session: int = Field(
        ge=15,
        le=120,
    )

    constants: list[str] = Field(
        min_length=2,
        max_length=5,
    )

    observations: list[str] = Field(
        min_length=3,
        max_length=6,
    )

    limitations: str = Field(
        min_length=8,
        max_length=300,
    )

    confirmation_question: str = Field(
        min_length=5,
        max_length=200,
    )


# ============================================
# 3. 自动从Pydantic生成Ollama JSON Schema
# ============================================

EXPERIMENT_SCHEMA: dict[str, Any] = (
    _GeneratedContent.model_json_schema()
)

EXPERIMENT_SCHEMA[
    "additionalProperties"
] = False


# ============================================
# 4. 保留具体错误原因
# ============================================

def _short_error(
    exc: Exception,
) -> str:
    """
    把Pydantic校验错误转换为易定位的字段信息。

    示例：
    task_b: Field required
    observations: List should have at least 3 items
    """

    if isinstance(
        exc,
        ValidationError,
    ):

        fragments = []

        for item in exc.errors(
            include_url=False
        )[:5]:

            field = ".".join(
                str(x)
                for x in item["loc"]
            ) or "根对象"

            fragments.append(
                f"{field}: {item['msg']}"
            )

        return "；".join(
            fragments
        )[:450]

    return str(exc)[:450]


# ============================================
# 5. 实践设计器：由Gemma 3生成具体方案
# ============================================

class ExperimentDesigner:

    SYSTEM = """
你是GrowthMentor-Agent的成长实践设计器。

仅输出符合JSON Schema的一个JSON对象。
不要输出Markdown、代码围栏或额外文字。

【任务】

用户已经明确选择方向A与方向B。
不需要你重新命名这两个方向。

只生成JSON Schema要求的11个内容字段。
不要输出option_a和option_b。

围绕用户已经熟悉或愿意尝试的
同一个问题，设计两个可比较的实践。

A和B必须是真正不同的活动，
并且具有不同的交付物。

尽量保持任务规模与时间预算接近。

【输出要求】

使用简洁中文。

question、common_problem、
task_a、task_b分别写一到两句话。

task_a和task_b必须描述具体行动，
不能只写“体验A”或“学习B”。

constants必须提供2到5条控制条件。

observations必须提供3到6条
能够由用户实际记录的观察指标。

minutes_per_session必须是
15到120之间的整数。

如果用户尚未确认时间预算，
给出可调整的建议，不要声称用户同意。

limitations要说明实验可能受到
任务难度、顺序、疲劳和熟悉程度影响。

不能通过短期体验判定固定天赋。

confirmation_question要邀请用户
确认共同任务和投入时间。

【重要边界】

不要虚构用户过去的经历。
不要擅自将方向A或B换成其他领域。

这份方案需要用户确认，
不能声称用户已经执行。
"""

    def __init__(
        self,
        model,
        attempts: int = 3,
    ):

        if (
            type(attempts) is not int
            or not 1 <= attempts <= 4
        ):
            raise ValueError(
                "attempts必须为1到4之间的整数"
            )

        self.model = model
        self.attempts = attempts

    # ========================================
    # 6. 对外接口保持不变
    # ========================================

    def build(
        self,
        goal: str,
        history: list[dict],
        draft: str,
        options: list[str],
    ) -> ComparisonProtocol:

        # ------------------------------------
        # 先检查用户输入
        # ------------------------------------

        if (
            not isinstance(options, list)
            or len(options) != 2
            or any(
                not isinstance(x, str)
                or len(x.strip()) < 2
                for x in options
            )
            or (
                options[0].strip()
                == options[1].strip()
            )
        ):
            raise ValueError(
                "必须提供两个不同且非空的比较方向"
            )

        # 用户输入才是选项名称的唯一来源
        user_a, user_b = (
            x.strip()
            for x in options
        )

        context: dict[str, Any] = {
            "goal": goal,

            "recent_dialogue": (
                history[-6:]
            ),

            "user_request": draft,

            "fixed_option_a": user_a,

            "fixed_option_b": user_b,

            "instructions": (
                "请仅生成实践计划内容，"
                "两个选项名称由程序"
                "按照用户输入原样保存"
            ),
        }

        failures: list[str] = []

        # ====================================
        # 7. LLM有限次生成与修正
        # ====================================

        for attempt in range(
            1,
            self.attempts + 1,
        ):

            if failures:

                context[
                    "上一轮格式错误"
                ] = failures[-1]

            try:

                # 真正调用Gemma 3
                raw = self.model.generate_json(
                    self.SYSTEM,

                    json.dumps(
                        context,
                        ensure_ascii=False,
                    ),

                    EXPERIMENT_SCHEMA,
                )

                if not isinstance(raw, dict):

                    raise ValueError(
                        "模型必须返回JSON对象"
                    )

                # --------------------------------
                # 兼容旧版测试的完整13字段输出
                # 但禁止模型擅自改变用户选项
                # --------------------------------

                for key, expected in (
                    ("option_a", user_a),
                    ("option_b", user_b),
                ):

                    if key in raw:

                        if (
                            not isinstance(
                                raw[key],
                                str,
                            )
                            or (
                                raw[key].strip()
                                != expected
                            )
                        ):

                            raise ValueError(
                                f"{key}被模型改动，"
                                "必须保持用户原始选项"
                            )

                # 新版模型原则上只返回11个内容字段
                content_only = {
                    key: value
                    for key, value in raw.items()

                    if key not in (
                        "option_a",
                        "option_b",
                    )
                }

                # --------------------------------
                # 第一次校验：
                # 内容结构是否合法
                # --------------------------------

                generated = (
                    _GeneratedContent.model_validate(
                        content_only
                    )
                )

                # --------------------------------
                # 第二次校验：
                # 合并用户选项后是否为完整方案
                # --------------------------------

                result = (
                    ComparisonProtocol.model_validate({
                        "option_a": user_a,

                        "option_b": user_b,

                        **generated.model_dump(),
                    })
                )

                return result

            except (
                ValueError,
                TypeError,
            ) as exc:

                failures.append(
                    f"第{attempt}次："
                    + _short_error(exc)
                )

        # ====================================
        # 8. 明确显示最终的失败原因
        # ====================================

        raise ValueError(
            "无法生成通过结构检查的"
            "对照实践计划。"
            + " | ".join(failures)
        )


# ============================================
# 9. 将结构化方案转换为自然语言
# ============================================

def render_protocol(
    plan: ComparisonProtocol,
) -> str:

    controls = "\n".join(
        f"- {item}"
        for item in plan.constants
    )

    observations = "\n".join(
        f"- {item}"
        for item in plan.observations
    )

    return (
        "**待你确认的一周对照实践方案**\n\n"

        f"**想回答的问题：** "
        f"{plan.question}\n\n"

        f"**共同问题/场景：** "
        f"{plan.common_problem}\n\n"

        f"**A｜{plan.option_a}**："
        f"{plan.task_a}\n"

        f"- 交付物：{plan.output_a}\n\n"

        f"**B｜{plan.option_b}**："
        f"{plan.task_b}\n"

        f"- 交付物：{plan.output_b}\n\n"

        "**建议时间：** "
        f"每次{plan.minutes_per_session}分钟，"
        "可按实际情况调整。"
        "一周内安排四次短实践"
        " A→B→B→A，"
        "结束后再比较。\n\n"

        f"**尽量保持一致：**\n"
        f"{controls}\n\n"

        f"**每次独立记录：**\n"
        f"{observations}\n\n"

        f"**判断限制：** "
        f"{plan.limitations}\n\n"

        f"**先确认：** "
        f"{plan.confirmation_question}\n\n"

        "*这只是未执行的方案，"
        "需要用户确认后才能开始。*"
    )
