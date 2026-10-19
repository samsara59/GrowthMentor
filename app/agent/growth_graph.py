
"""GrowthMentor 的 LangGraph 可恢复工作流。

本课实现：
1. 节点状态
2. 用户确认中断
3. 流式执行事件
4. SQLite Checkpointer
5. 复用现有 AdaptiveMentor
"""

from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import (
    StateGraph,
    START,
    END,
)
from langgraph.types import interrupt

from app.agent.adaptive_mentor import (
    ChatEntry,
    MentorState,
)
from app.domain.contracts import GrowthTask


# ============================================
# 1. 工作流共享状态
# ============================================

class GrowthGraphState(
    TypedDict,
    total=False,
):
    # 本轮用户输入
    question: str
    goal: str
    program_id: str

    # 已有成长信息
    history: list[dict]
    confirmed_notes: list[dict]

    # 工作流内部状态
    preview: str
    approved: bool
    phase: str

    # 最终回答和执行记录
    result: str
    action: str
    focus: str | None
    trace: list[str]
    sources: list[dict]


# ============================================
# 2. 发送实时进度事件
# ============================================

def emit(
    stage: str,
    message: str,
) -> None:
    """
    stream_mode包含custom时，
    事件会发送到Streamlit。

    在普通invoke测试中，
    没有流式监听器时直接跳过。
    """

    try:
        writer = get_stream_writer()
    except (RuntimeError, LookupError):
        return

    writer({
        "stage": stage,
        "message": message,
    })


# ============================================
# 3. 构建Graph，复用现有导师
# ============================================

def build_growth_graph(
    mentor,
    checkpointer,
):
    """
    mentor：
        现有AdaptiveMentor，
        必须有reply(state, text)。

    checkpointer：
        SqliteSaver等LangGraph检查点存储。
    """

    # ========================================
    # 节点A：准备输入
    # ========================================

    def prepare(
        state: GrowthGraphState,
    ) -> dict:

        question = (
            state.get("question") or ""
        ).strip()

        goal = (
            state.get("goal") or ""
        ).strip()

        if not question:
            raise ValueError(
                "成长问题不能为空"
            )

        if not goal:
            raise ValueError(
                "成长目标不能为空"
            )

        emit(
            "prepare",
            "已检查问题和成长目标",
        )

        return {
            "question": question,
            "goal": goal,
            "phase": "awaiting_confirmation",

            "preview": (
                "准备使用本地 Gemma 3 "
                "分析以下成长问题："
                + question
            ),

            # 新一轮开始，清理旧的结果
            "approved": False,
            "result": "",
            "action": "",
            "focus": None,
            "trace": [],
            "sources": [],
        }

    # ========================================
    # 节点B：暂停等待用户确认
    # ========================================

    def approval(
        state: GrowthGraphState,
    ) -> dict:

        # 重要：interrupt()之前
        # 不执行不可重复的外部写入。
        #
        # 恢复时该节点会从头再次运行。

        answer = interrupt({
            "type": "approval",
            "question": state["question"],
            "message": (
                "是否允许本轮调用"
                "本机 Gemma 3？"
            ),
        })

        approved = (
            answer is True
        )

        emit(
            "approval",
            (
                "用户已确认"
                if approved
                else "用户拒绝执行"
            ),
        )

        return {
            "approved": approved,
            "phase": (
                "approved"
                if approved
                else "declined"
            ),
        }

    # ========================================
    # 节点C：真正调用已有AdaptiveMentor
    # ========================================

    def mentor_turn(
        state: GrowthGraphState,
    ) -> dict:

        emit(
            "mentor_turn",
            "开始调用现有 AdaptiveMentor",
        )

        task = GrowthTask(
            task_id="langgraph74",
            program_id=state.get(
                "program_id",
                "self_discovery",
            ),
            goal=state["goal"],
        )

        old_history = [
            ChatEntry.model_validate(item)
            for item in state.get(
                "history",
                [],
            )
        ]

        mentor_state = MentorState(
            task=task,
            history=old_history,
            confirmed_notes=state.get(
                "confirmed_notes",
                [],
            ),
        )

        # 这里是真正的Gemma 3导师，
        # 不是模拟回答。
        #
        # 内部仍可能调用：
        # knowledge.retrieve
        # ExperimentDesigner等已有模块。

        new_state, turn = mentor.reply(
            mentor_state,
            state["question"],
        )

        emit(
            "mentor_turn",
            "AdaptiveMentor 已返回结果",
        )

        return {
            "history": [
                msg.model_dump()
                for msg in new_state.history
            ],

            "result": turn.message,
            "action": turn.action,
            "focus": turn.focus,

            "trace": list(turn.trace),
            "sources": list(turn.sources),

            "phase": "completed",
        }

    # ========================================
    # 节点D：用户拒绝后结束
    # ========================================

    def cancelled(
        state: GrowthGraphState,
    ) -> dict:

        emit(
            "cancelled",
            "本轮取消，没有调用导师",
        )

        return {
            "phase": "cancelled",
            "result": (
                "你取消了本轮执行；"
                "没有调用本机 Gemma 3。"
            ),
            "action": "cancelled",
        }

    # ========================================
    # 4. 根据用户确认结果选择路线
    # ========================================

    def route_after_approval(
        state: GrowthGraphState,
    ) -> str:

        return (
            "mentor_turn"
            if state.get("approved", False)
            else "cancelled"
        )

    # ========================================
    # 5. 注册所有Graph节点
    # ========================================

    builder = StateGraph(
        GrowthGraphState
    )

    builder.add_node(
        "prepare",
        prepare,
    )

    builder.add_node(
        "approval",
        approval,
    )

    builder.add_node(
        "mentor_turn",
        mentor_turn,
    )

    builder.add_node(
        "cancelled",
        cancelled,
    )

    # ========================================
    # 6. 设置执行边
    # ========================================

    builder.add_edge(
        START,
        "prepare",
    )

    builder.add_edge(
        "prepare",
        "approval",
    )

    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "mentor_turn": "mentor_turn",
            "cancelled": "cancelled",
        },
    )

    builder.add_edge(
        "mentor_turn",
        END,
    )

    builder.add_edge(
        "cancelled",
        END,
    )

    # ========================================
    # 7. 连接持久化Checkpointer
    # ========================================

    graph = builder.compile(
        checkpointer=checkpointer
    )

    return graph
