
import re

from typing import Sequence

from app.memory.schema import AgentSession


# 与第58课 graph_workflow.py 对应的节点
NODES = {
    "begin": "用户输入",
    "memory_load": "Memory Load",
    "supervisor": "Supervisor",
    "rag_agent": "RAG Agent",
    "interview_agent": "Interview Agent",
    "assessment_agent": "Assessment Agent",
    "verification_agent": "Verification Agent",
    "memory_agent": "Memory Agent",
    "unsupported": "Unsupported",
    "memory_save": "Memory Save",
    "finish": "结束",
}


EDGES = [
    ("begin", "memory_load"),
    ("memory_load", "supervisor"),

    ("supervisor", "rag_agent"),
    ("supervisor", "interview_agent"),
    ("supervisor", "assessment_agent"),
    ("supervisor", "verification_agent"),
    ("supervisor", "memory_agent"),
    ("supervisor", "unsupported"),

    ("rag_agent", "memory_save"),
    ("interview_agent", "memory_save"),
    ("assessment_agent", "memory_save"),
    ("verification_agent", "memory_save"),
    ("memory_agent", "memory_save"),
    ("unsupported", "memory_save"),

    ("memory_save", "finish"),
]


def validate_session_id(value: str) -> str:
    """限制本地会话ID格式，避免误用路径等内容。"""

    value = value.strip()

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}",
        value,
    ):
        raise ValueError(
            "会话ID只能包含英文字母、数字、"
            "下划线和连字符，长度为1至40。"
        )

    return value


def format_trace(
    path: Sequence[str],
) -> str:
    """将实际执行节点转换成人类可读的路径。"""

    if not path:
        return "尚未执行工作流"

    return " → ".join(
        NODES.get(node, node)
        for node in path
    )


def build_graphviz_dot(
    path: Sequence[str] = (),
) -> str:
    """
    构造Graphviz DOT可视化图。

    绿色：最近一次执行经过的节点与连接。
    灰色：未经过的节点。
    """

    visited = set(path)

    actual_path = [
        "begin",
        *path,
        "finish",
    ] if path else []

    visited_edges = set(
        zip(actual_path, actual_path[1:])
    )

    lines = [
        "digraph GrowthMentor {",
        'graph [rankdir=TB, bgcolor="transparent", '
        'nodesep=0.22, ranksep=0.32, pad=0.15];',

        'node [shape=box, style="rounded,filled", '
        'fontname="Arial", fontsize=10, margin="0.12,0.08"];',

        'edge [arrowsize=0.55];',
    ]

    for node, label in NODES.items():

        highlighted = node in visited

        if node in {"begin", "finish"}:
            highlighted = bool(path)

        background = (
            "#DFF3E7"
            if highlighted
            else "#F1F3F6"
        )

        border = (
            "#28976B"
            if highlighted
            else "#C8CDD5"
        )

        lines.append(
            f'"{node}" [label="{label}", '
            f'fillcolor="{background}", '
            f'color="{border}"];'
        )

    for source, target in EDGES:

        highlighted = (
            source,
            target,
        ) in visited_edges

        color = (
            "#28976B"
            if highlighted
            else "#CCD2DB"
        )

        width = 2.3 if highlighted else 1.0

        lines.append(
            f'"{source}" -> "{target}" '
            f'[color="{color}", penwidth={width}];'
        )

    lines.append("}")

    return "\n".join(lines)


def stage_description(
    session: AgentSession,
) -> str:
    """根据当前状态决定应该提醒用户什么。"""

    if session.stage == "awaiting_situation":
        return (
            "请描述经历的情境："
            "什么时候、发生了什么、遇到了什么问题？"
        )

    if session.stage == "awaiting_action":
        return (
            "请描述你本人采取的行动："
            "具体做了什么、采用哪些方法？"
        )

    if session.stage == "awaiting_outcome":
        return (
            "请描述最后的结果："
            "成功、部分成功、失败或暂时未知均可。"
        )

    if session.stage == "awaiting_verification":

        state = session.verification

        if state is None:
            return "验证状态不完整，请检查会话记录。"

        if state.pending_answer is not None:
            return (
                "上一条回答已保存，但尚未成功分析。"
                "请执行 /retryverify。"
            )

        return state.pending_question

    return (
        "当前没有正在进行的任务。"
        "可以选择开始优势探索或查询教材知识。"
    )


def get_summary(
    session: AgentSession,
) -> dict:
    """生成控制台顶部统计信息。"""

    return {
        "experiences": len(session.experiences),

        "assessments": sum(
            item is not None
            for item in session.assessments
        ),

        "candidates": (
            len(session.cross_report.candidates)
            if session.cross_report is not None
            else 0
        ),

        "verification_rounds": (
            len(session.verification.turns)
            if session.verification is not None
            else 0
        ),

        "stage": session.stage,
    }
