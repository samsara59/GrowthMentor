
from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from app.agent.orchestrator import (
    AgentOrchestrator,
    AgentReply,
)

from app.memory.schema import (
    AgentSession,
)

from app.memory.store import MemoryStore


# ============================================
# 1. LangGraph共享状态
# ============================================

class GrowthGraphState(TypedDict, total=False):
    message: str
    session_id: str

    session: AgentSession

    route: str
    reply: AgentReply

    # 记录这轮经过哪些图节点
    path: list[str]

    # 删除整个会话时，不再重新创建数据库记录
    skip_save: bool


def add_path(state: GrowthGraphState, name: str):
    return state.get("path", []) + [name]


# ============================================
# 2. 构造LangGraph
# ============================================

def build_workflow(
    agent: AgentOrchestrator | None = None,
    memory: MemoryStore | None = None,
):
    """
    组装GrowthMentor-Agent图。

    agent和memory可为None，仅用于导出图结构。
    真正执行graph.invoke时必须提供二者。
    """

    def require_services():
        if agent is None or memory is None:
            raise RuntimeError(
                "执行工作流需要Agent和MemoryStore"
            )
        return agent, memory

    # ========================================
    # 3. Memory Load
    # ========================================

    def memory_load(state: GrowthGraphState):
        _, store = require_services()

        session = store.load_session(
            state["session_id"]
        )

        return {
            "session": session,
            "path": ["memory_load"],
            "skip_save": False,
        }

    # ========================================
    # 4. Supervisor
    # ========================================

    def supervisor(state: GrowthGraphState):
        current_agent, _ = require_services()

        message = state["message"].strip()
        session = state["session"]

        if not message:
            raise ValueError("用户消息不能为空")

        command = message.split()[0]

        # A. 显式业务命令
        if command == "/ask":
            route = "rag"

        elif command == "/start":
            route = "interview"

        elif command in {"/compare", "/reanalyze"}:
            route = "assessment"

        elif command in {
            "/verify",
            "/retryverify",
            "/cancelverify",
        }:
            route = "verification"

        elif command in {
            "/help",
            "/status",
            "/history",
            "/verification",
            "/save",
            "/reset",
            "/edit",
            "/delete",
            "/forget",
        }:
            route = "memory"

        elif message.startswith("/"):
            route = "other"

        # B. 正在执行的任务优先
        elif session.stage == "awaiting_verification":
            route = "verification"

        elif session.stage in {
            "awaiting_situation",
            "awaiting_action",
            "awaiting_outcome",
        }:
            route = "interview"

        # C. 空闲状态下才调用LLM Router
        else:
            decision = current_agent.router.route(
                message=message,
                stage=session.stage,
            )

            mapping = {
                "knowledge": "rag",
                "start_exploration": "interview",
                "experience_reply": "other",
                "other": "other",
            }

            route = mapping.get(
                decision.action,
                "other",
            )

        return {
            "route": route,
            "path": add_path(state, "supervisor"),
        }

    # ========================================
    # 5. RAG Agent
    # ========================================

    def rag_agent(state: GrowthGraphState):
        current_agent, _ = require_services()

        message = state["message"]
        session = state["session"]

        # 自然语言知识请求，转换成显式命令
        # 防止再次调用Router造成双重分类
        if not message.startswith("/ask"):
            message = "/ask " + message

        reply = current_agent.handle(
            message,
            session,
        )

        return {
            "session": session,
            "reply": reply,
            "path": add_path(state, "rag_agent"),
        }

    # ========================================
    # 6. Interview Agent
    # ========================================

    def interview_agent(state: GrowthGraphState):
        current_agent, _ = require_services()

        message = state["message"]
        session = state["session"]

        # 空闲时Router选择了优势探索
        if session.stage == "idle":
            message = "/start"

        # 正在访谈时传入真实回答
        reply = current_agent.handle(
            message,
            session,
        )

        return {
            "session": session,
            "reply": reply,
            "path": add_path(state, "interview_agent"),
        }

    # ========================================
    # 7. Assessment Agent
    # ========================================

    def assessment_agent(state: GrowthGraphState):
        current_agent, _ = require_services()

        message = state["message"]
        session = state["session"]

        if message == "/compare":
            reply = current_agent.compare_experiences(
                session
            )

        elif message.startswith("/reanalyze"):
            parts = message.split()

            if (
                len(parts) != 2
                or parts[0] != "/reanalyze"
            ):
                reply = AgentReply(
                    "用法：/reanalyze 1",
                    "question",
                )
            else:
                try:
                    number = int(parts[1])

                    reply = (
                        current_agent.reanalyze_experience(
                            number,
                            session,
                        )
                    )

                except ValueError as exc:
                    reply = AgentReply(
                        f"命令错误：{exc}",
                        "question",
                    )
        else:
            reply = AgentReply(
                "无法识别Assessment操作。",
                "unsupported",
            )

        return {
            "session": session,
            "reply": reply,
            "path": add_path(state, "assessment_agent"),
        }

    # ========================================
    # 8. Verification Agent
    # ========================================

    def verification_agent(state: GrowthGraphState):
        current_agent, _ = require_services()

        session = state["session"]

        reply = current_agent.handle(
            state["message"],
            session,
        )

        return {
            "session": session,
            "reply": reply,
            "path": add_path(state, "verification_agent"),
        }

    # ========================================
    # 9. Memory Agent
    # ========================================

    def memory_agent(state: GrowthGraphState):
        _, store = require_services()

        message = state["message"]
        session_id = state["session_id"]
        session = state["session"]

        parts = message.split()
        command = parts[0]

        skip_save = False

        if command == "/help":
            answer = (
                "可用命令：\n"
                "/start  /ask 问题  /compare\n"
                "/verify 1  /retryverify  /cancelverify\n"
                "/verification  /history  /status\n"
                "/reanalyze 1  /save  /reset\n"
                "/edit 1 action 新内容\n"
                "/delete 1  /delete 1 YES\n"
                "/forget  /forget YES"
            )

        elif command == "/status":
            answer = (
                f"会话：{session_id}\n"
                f"阶段：{session.stage}\n"
                f"已完成经历：{len(session.experiences)}\n"
                f"已完成单经历分析："
                f"{sum(x is not None for x in session.assessments)}\n"
                f"跨经历报告："
                f"{'已生成' if session.cross_report else '暂无'}\n"
                f"验证轮数："
                f"{len(session.verification.turns) if session.verification else 0}\n"
                f"当前草稿：{list(session.draft)}"
            )

        elif command == "/history":
            lines = ["【历史经历】"]

            if not session.experiences:
                lines.append("目前没有已完成的经历。")

            for i, record in enumerate(
                session.experiences,
                1,
            ):
                analysis = session.assessments[i - 1]

                lines.extend([
                    f"\n经历{i}",
                    f"情境：{record.situation}",
                    f"行动：{record.action}",
                    f"结果：{record.outcome}",
                    (
                        "分析："
                        + (
                            "已保存"
                            if analysis is not None
                            else "尚无有效分析"
                        )
                    ),
                ])

                if analysis is not None:
                    names = [
                        x.name
                        for x in analysis.hypotheses
                    ]
                    lines.append(
                        "候选方向："
                        + ("、".join(names) or "暂无")
                    )

            answer = "\n".join(lines)

        elif command == "/verification":
            verification = session.verification

            if verification is None:
                answer = "目前没有假设验证记录。"
            else:
                lines = [
                    f"候选方向：{verification.candidate_name}",
                    f"已完成轮数：{len(verification.turns)}",
                ]

                for i, turn in enumerate(
                    verification.turns,
                    1,
                ):
                    lines.extend([
                        f"\n第{i}轮：{turn.question}",
                        f"回答：{turn.answer}",
                        f"判断：{turn.relation}",
                        f"原文：{turn.quote}",
                        f"原因：{turn.reason}",
                    ])

                if verification.pending_answer:
                    lines.append(
                        "存在未分析回答，请输入 /retryverify。"
                    )

                elif session.stage == "awaiting_verification":
                    lines.append(
                        "当前问题："
                        + verification.pending_question
                    )

                answer = "\n".join(lines)

        elif command == "/save":
            answer = "当前会话将保存到本地SQLite。"

        elif command == "/reset":
            session.stage = "idle"
            session.draft.clear()
            session.verification = None
            answer = (
                "已放弃当前未完成的任务。"
                "历史经历和跨经历报告保留。"
            )

        elif command == "/edit":
            args = message.split(maxsplit=3)

            if len(args) != 4:
                answer = (
                    "用法：/edit 1 action 更正后的真实描述"
                )
            else:
                try:
                    session = store.update_experience(
                        session_id,
                        session,
                        int(args[1]),
                        args[2],
                        args[3],
                    )
                    answer = (
                        "经历已更正。"
                        "依赖旧事实的分析和验证已失效。"
                    )
                except ValueError as exc:
                    answer = f"更正失败：{exc}"

        elif command == "/delete":
            if len(parts) == 2:
                answer = (
                    f"请确认删除经历{parts[1]}。\n"
                    f"确认命令：/delete {parts[1]} YES\n"
                    "不输入确认命令就不会删除。"
                )
            elif len(parts) == 3 and parts[2] == "YES":
                try:
                    number = int(parts[1])
                    session = store.delete_experience(
                        session_id,
                        session,
                        number,
                    )
                    answer = f"经历{number}已删除。"
                except ValueError as exc:
                    answer = f"删除失败：{exc}"
            else:
                answer = "用法：/delete 1，然后 /delete 1 YES"

        elif command == "/forget":
            if message == "/forget YES":
                store.forget_session(session_id)
                session = AgentSession()
                skip_save = True
                answer = (
                    "当前会话的SQLite记录已删除。"
                    "应用层删除不等于安全擦除或删除备份。"
                )
            else:
                answer = (
                    "这会清除当前会话全部持久化记忆。\n"
                    "确认请输入：/forget YES"
                )

        else:
            answer = "未知记忆命令。"

        return {
            "session": session,
            "reply": AgentReply(
                answer,
                "memory",
            ),
            "skip_save": skip_save,
            "path": add_path(state, "memory_agent"),
        }

    # ========================================
    # 10. Unsupported
    # ========================================

    def unsupported(state: GrowthGraphState):
        return {
            "reply": AgentReply(
                "当前请求没有对应的已授权业务功能。",
                "unsupported",
            ),
            "path": add_path(state, "unsupported"),
        }

    # ========================================
    # 11. Memory Save
    # ========================================

    def memory_save(state: GrowthGraphState):
        _, store = require_services()

        if not state.get("skip_save", False):
            store.save_session(
                state["session_id"],
                state["session"],
            )

        return {
            "path": add_path(state, "memory_save"),
        }

    # ========================================
    # 12. 注册所有节点
    # ========================================

    builder = StateGraph(GrowthGraphState)

    builder.add_node("memory_load", memory_load)
    builder.add_node("supervisor", supervisor)

    builder.add_node("rag_agent", rag_agent)
    builder.add_node("interview_agent", interview_agent)
    builder.add_node("assessment_agent", assessment_agent)
    builder.add_node("verification_agent", verification_agent)
    builder.add_node("memory_agent", memory_agent)
    builder.add_node("unsupported", unsupported)

    builder.add_node("memory_save", memory_save)

    # ========================================
    # 13. 连接图
    # ========================================

    builder.add_edge(START, "memory_load")
    builder.add_edge("memory_load", "supervisor")

    # 根据Supervisor输出选择执行分支
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        {
            "rag": "rag_agent",
            "interview": "interview_agent",
            "assessment": "assessment_agent",
            "verification": "verification_agent",
            "memory": "memory_agent",
            "other": "unsupported",
        },
    )

    # 所有业务节点结束后统一保存
    for node in (
        "rag_agent",
        "interview_agent",
        "assessment_agent",
        "verification_agent",
        "memory_agent",
        "unsupported",
    ):
        builder.add_edge(node, "memory_save")

    builder.add_edge("memory_save", END)

    return builder.compile()
