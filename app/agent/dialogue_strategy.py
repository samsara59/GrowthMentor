
"""Growth Programs 的对话原则，不预设固定提问顺序。"""

from collections.abc import Mapping


GUIDES = {
    "self_discovery": (
        "当前任务是自我理解，不是优势分类。"
        "分清三种互不等价的线索："
        "WHAT=用户自发关注或喜欢探索的事；"
        "HOW=在具体任务中自然采用的行为方式，"
        "不能由成绩直接推出；"
        "WHY=用户重视的价值与意义，"
        "不能由职业或任务直接推出。"
        "根据用户提问和已有证据自由选择"
        "最需要理解的方向，"
        "不按 WHAT→HOW→WHY 的固定顺序进行。"
        "对兴趣、能力、价值观的解释只能是暂时假设，"
        "必须允许反例和用户修正。"
    ),

    "reflective_decision": (
        "帮助用户清楚区分现实事实、个人判断、"
        "备选方案、约束条件与实际后果。"
        "不要代替用户作决定；"
        "若涉及特定哲学著作的原文观点，"
        "须先有检索证据。"
    ),

    "historical_learning": (
        "帮助用户从可核实的历史人物经历"
        "提炼可检验的思考方法。"
        "区分史实、人物自己的说法和你的推断；"
        "没有来源时不要编造事迹或言论。"
        "不要将过去情境的做法机械套用到用户身上。"
    ),
}


FOCUS_LABELS = {
    "what": "喜欢的事（WHAT）",
    "how": "自然擅长的方式（HOW）",
    "why": "重视的事（WHY）",
    "decision": "现实选择与权衡",
    "practice": "行动与反馈",
    "general": "一般成长对话",
}


NOTE_GROUPS = (
    ("interest", "WHAT · 兴趣线索"),
    ("strength", "HOW · 行为与能力证据"),
    ("value", "WHY · 价值观"),
    ("goal", "长期目标"),
    ("experiment", "行动与实践反馈"),
    ("other", "其他确认信息"),
)


def guide_for(program_id: str) -> str:
    """未知 Program 使用通用原则，而非套用自我发现问卷。"""

    return GUIDES.get(
        program_id,
        "根据用户设定的 Growth Program 目标，"
        "识别当前信息缺口，"
        "自由选择追问、检索、总结或小规模实践建议。"
        "不要强行使用自我发现的三维探索框架。",
    )


def build_growth_view(
    notes: list[Mapping],
) -> list[dict]:
    """
    将用户确认的笔记组织成成长视图。

    不生成标签、不推断天赋，
    不把模型猜测当成用户事实。
    """

    groups = []

    for category, title in NOTE_GROUPS:

        entries = []

        for note in notes:

            if note.get("category") != category:
                continue

            content = note.get(
                "content",
                note.get("text", ""),
            )

            if (
                isinstance(content, str)
                and content.strip()
            ):
                entries.append(
                    content.strip()
                )

        groups.append({
            "category": category,
            "title": title,
            "entries": entries,
        })

    return groups
