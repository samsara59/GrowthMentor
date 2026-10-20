
import sys
from pathlib import Path

import streamlit as st


# ============================================
# 1. 定位项目根目录
# ============================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================
# 2. 导入后端模块
# ============================================

from app.agent.graph_workflow import build_workflow

from app.agent.guidance import (
    GUIDANCE,
    explain_choice,
)

from app.ui.live_monitor import (
    stream_turn,
    next_expected_node,
)

from app.ui.presenter import (
    build_graphviz_dot,
    format_trace,
    get_summary,
    stage_description,
    validate_session_id,
)

from app.memory.store import MemoryStore

from scripts.chat_agent import build_agent


MEMORY_DB = ROOT / "data" / "user_memory.db"


# ============================================
# 3. Streamlit页面设置
# ============================================

st.set_page_config(
    page_title="GrowthMentor-Agent",
    page_icon="🧭",
    layout="wide",
)


# ============================================
# 4. 初始化服务
# ============================================

@st.cache_resource(show_spinner=False)
def get_services():

    memory = MemoryStore(MEMORY_DB)

    agent = build_agent()

    graph = build_workflow(
        agent=agent,
        memory=memory,
    )

    return memory, graph


# ============================================
# 5. 初始化页面状态
# ============================================

def init_state():

    defaults = {
        "session_id": "lesson54_demo",
        "session_id_input": "lesson54_demo",

        "messages": [],
        "last_path": [],
        "last_timeline": [],

        "guidance_hint": "",

        # 本课新增：
        # 等待LangGraph执行的一条命令
        "pending_command": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# ============================================
# 6. 将用户命令放入等待队列
# ============================================

def send(
    message: str,
    graph,
):

    message = message.strip()

    if not message:
        return

    if (
        st.session_state.get("pending_command")
        is not None
    ):

        st.warning(
            "上一条命令尚未处理，请勿重复提交。"
        )

        return

    st.session_state["pending_command"] = message

    st.session_state["messages"].append({
        "role": "user",
        "content": message,
    })

    st.session_state["guidance_hint"] = ""


# ============================================
# 7. 执行等待中的命令
# ============================================

def run_pending_command(
    graph,
    graph_slot,
    path_slot,
    status_slot,
    timeline_slot,
):

    message = st.session_state.get(
        "pending_command"
    )

    if message is None:
        return

    # 先清除待执行状态
    # 防止页面重绘时重复执行写入操作
    st.session_state["pending_command"] = None

    last = None
    history = []

    status_slot.info(
        "工作流已启动，等待Memory Load返回……"
    )

    # ========================================
    # 8. 按节点读取流式结果
    # ========================================

    try:

        for progress in stream_turn(
            graph,
            message,
            st.session_state["session_id"],
        ):

            last = progress

            # 记录已完成节点和累计耗时
            history.append(
                f"{len(history) + 1}. "
                f"{progress.node} "
                f"（累计{progress.elapsed_seconds:.2f}秒）"
            )

            # --------------------------------
            # A. 更新右侧节点图
            # --------------------------------

            graph_slot.graphviz_chart(
                build_graphviz_dot(
                    progress.path
                ),
                width="stretch",
            )

            # --------------------------------
            # B. 更新执行路径
            # --------------------------------

            path_slot.code(
                format_trace(progress.path),
                language="text",
            )

            # --------------------------------
            # C. 更新执行时间线
            # --------------------------------

            timeline_slot.code(
                "\n".join(history),
                language="text",
            )

            # --------------------------------
            # D. 预测下一节点
            # --------------------------------

            upcoming = next_expected_node(
                progress.node,
                progress.route,
            )

            if upcoming is None:

                status_slot.success(
                    "本轮工作流节点已全部完成。"
                )

            else:

                status_slot.info(
                    f"已完成：{progress.node}\n\n"
                    f"预计下一节点：{upcoming}\n\n"
                    "注意：尚未收到该节点的完成事件。"
                )

        # ====================================
        # 9. 检查最终回复
        # ====================================

        if last is None or last.reply is None:

            raise RuntimeError(
                "工作流没有产生完整结果"
            )

        reply = last.reply

        # ====================================
        # 10. 保存本次浏览器展示数据
        # ====================================

        st.session_state["messages"].append({
            "role": "assistant",
            "content": reply.message,
            "mode": reply.mode,
            "status": reply.status,
            "citations": [
                {
                    "id": citation.evidence_id,
                    "source": citation.source,
                    "location": citation.location,
                    "quote": citation.quote,
                }
                for citation in reply.citations
            ],
        })

        st.session_state["last_path"] = list(
            last.path
        )

        st.session_state["last_timeline"] = (
            history
        )

    # ========================================
    # 11. 捕获运行失败
    # ========================================

    except Exception as exc:

        st.session_state["last_path"] = (
            list(last.path)
            if last is not None
            else []
        )

        st.session_state["last_timeline"] = (
            history
        )

        st.session_state["messages"].append({
            "role": "assistant",
            "content": (
                "工作流没有确认完整成功："
                f"{type(exc).__name__}: {exc}\n\n"
                "部分业务节点可能已经产生副作用。"
                "请先检查会话状态，"
                "不要直接重复提交删除或修改命令。"
            ),
            "mode": "error",
            "status": None,
            "citations": [],
        })

        status_slot.error(
            "执行失败，请检查当前会话状态。"
        )

    # 刷新正式的用户状态和聊天窗口
    st.rerun()


# ============================================
# 12. 显示聊天历史
# ============================================

def show_messages():

    messages = st.session_state["messages"]

    if not messages:

        st.info(
            "这里显示当前浏览器会话的聊天。"
            "已保存的经历请在下方标签页查看。"
        )

    for item in messages:

        with st.chat_message(item["role"]):

            st.write(item["content"])

            if item.get("mode"):
                st.caption(
                    f"处理模块：{item['mode']}"
                )

            if item.get("status"):
                st.caption(
                    f"回答状态：{item['status']}"
                )

            citations = item.get(
                "citations",
                [],
            )

            if citations:

                with st.expander(
                    f"教材引用（{len(citations)}条）"
                ):

                    for citation in citations:

                        st.write(
                            f"[{citation['id']}] "
                            f"{citation['source']}"
                        )

                        st.write(
                            "位置：",
                            citation["location"],
                        )

                        st.write(
                            "原文：",
                            citation["quote"],
                        )


# ============================================
# 13. 显示数字选项引导
# ============================================

def show_guidance(session):

    options = GUIDANCE.get(
        session.stage
    )

    if not options:
        return

    with st.expander(
        "不知道如何回答？选择引导方向",
        expanded=True,
    ):

        st.caption(
            "选择只展示提示，"
            "不会将示例自动保存为用户经历。"
        )

        for number, value in options.items():

            title, _ = value

            if st.button(
                f"{number}. {title}",
                key=(
                    f"hint_{session.stage}_{number}"
                ),
            ):

                st.session_state[
                    "guidance_hint"
                ] = explain_choice(
                    session.stage,
                    str(number),
                )

                st.rerun()

        if st.session_state["guidance_hint"]:

            st.info(
                st.session_state["guidance_hint"]
            )


# ============================================
# 14. 显示历史经历
# ============================================

def show_experiences(session, graph):

    if not session.experiences:

        st.info(
            "当前没有完整经历。"
            "可以点击左侧开始优势探索。"
        )

        return

    for number, record in enumerate(
        session.experiences,
        start=1,
    ):

        with st.expander(
            f"经历 {number}"
        ):

            st.markdown("**情境**")
            st.write(record.situation)

            st.markdown("**行动**")
            st.write(record.action)

            st.markdown("**结果**")
            st.write(record.outcome)

            assessment = (
                session.assessments[number - 1]
            )

            if assessment is None:

                st.warning(
                    "目前没有有效的单经历分析。"
                )

                if st.button(
                    "重新分析",
                    key=f"reanalyze_{number}",
                ):

                    send(
                        f"/reanalyze {number}",
                        graph,
                    )

                    st.rerun()

            else:

                st.markdown(
                    "**行为观察**"
                )

                for observation in assessment.observations:

                    st.write(
                        observation.description
                    )

                    st.caption(
                        f"原文：{observation.quote}"
                    )

                st.markdown(
                    "**待验证的优势方向**"
                )

                if not assessment.hypotheses:

                    st.write(
                        "暂无符合条件的候选方向。"
                    )

                for hypothesis in assessment.hypotheses:

                    st.write(
                        f"{hypothesis.name}："
                        f"{hypothesis.interpretation}"
                    )

                st.markdown(
                    "**下一轮建议问题**"
                )

                st.write(
                    assessment.follow_up_question
                )

                st.caption(
                    "分析限制："
                    + assessment.limitation
                )

            # =================================
            # 15. 更正经历
            # =================================

            with st.form(
                f"edit_form_{number}"
            ):

                field_name = st.selectbox(
                    "更正字段",
                    [
                        "situation",
                        "action",
                        "outcome",
                    ],
                    format_func=lambda x: {
                        "situation": "情境",
                        "action": "行动",
                        "outcome": "结果",
                    }[x],
                )

                new_value = st.text_area(
                    "更正后的真实描述",
                    height=90,
                )

                submitted = st.form_submit_button(
                    "保存更正"
                )

                if submitted:

                    if not new_value.strip():

                        st.error(
                            "更正内容不能为空。"
                        )

                    else:

                        send(
                            f"/edit {number} "
                            f"{field_name} "
                            f"{new_value.strip()}",
                            graph,
                        )

                        st.rerun()

            # =================================
            # 16. 确认删除经历
            # =================================

            confirm_key = (
                f"delete_confirm_"
                f"{st.session_state['session_id']}_"
                f"{len(session.experiences)}_"
                f"{number}"
            )

            approved = st.checkbox(
                "我确认删除这条经历和对应分析",
                key=confirm_key,
            )

            if st.button(
                f"删除经历 {number}",
                key=f"delete_button_{number}",
                disabled=not approved,
            ):

                send(
                    f"/delete {number} YES",
                    graph,
                )

                st.rerun()


# ============================================
# 17. 显示跨经历候选方向
# ============================================

def show_cross_report(session, graph):

    report = session.cross_report

    if report is None:

        st.info(
            "没有跨经历报告，"
            "请先完成至少两条经历。"
        )

        if st.button(
            "生成跨经历报告",
            key="compare_button",
        ):

            send(
                "/compare",
                graph,
            )

            st.rerun()

        return

    st.write(
        f"候选方向：{len(report.candidates)}个"
    )

    if not report.candidates:

        st.info(
            "当前没有足够的跨经历重复行为证据。"
        )

    for number, candidate in enumerate(
        report.candidates,
        start=1,
    ):

        with st.expander(
            f"候选 {number}：{candidate.name}",
            expanded=True,
        ):

            st.write(
                candidate.interpretation
            )

            st.markdown(
                "**支持证据**"
            )

            for evidence in candidate.supports:

                st.write(
                    f"经历{evidence.experience_number} "
                    f"({evidence.source_field})："
                    f"{evidence.quote}"
                )

            if candidate.possible_counterevidence:

                st.markdown(
                    "**可能的限制或反例线索**"
                )

                for evidence in (
                    candidate.possible_counterevidence
                ):

                    st.write(
                        f"经历{evidence.experience_number}："
                        f"{evidence.quote}"
                    )

            st.caption(
                "不确定性："
                + candidate.uncertainty
            )

            if session.stage == "idle":

                if st.button(
                    f"验证候选 {number}",
                    key=f"verify_{number}",
                ):

                    send(
                        f"/verify {number}",
                        graph,
                    )

                    st.rerun()

    st.markdown(
        "**建议问题**"
    )

    st.write(
        report.next_question
    )

    st.caption(
        "分析限制："
        + report.limitations
    )


# ============================================
# 18. 显示动态验证历史
# ============================================

def show_verification(session, graph):

    state = session.verification

    if state is None:

        st.info(
            "暂无验证记录。"
            "请先生成跨经历报告并选择候选方向。"
        )

        return

    st.markdown(
        f"**候选方向：{state.candidate_name}**"
    )

    st.write(
        f"已完成验证轮数：{len(state.turns)}"
    )

    for number, turn in enumerate(
        state.turns,
        start=1,
    ):

        with st.expander(
            f"验证第{number}轮",
            expanded=True,
        ):

            st.write(
                "问题：",
                turn.question,
            )

            st.write(
                "回答：",
                turn.answer,
            )

            st.write(
                "证据关系：",
                turn.relation,
            )

            st.write(
                "原文：",
                turn.quote or "无",
            )

            st.write(
                "判断理由：",
                turn.reason,
            )

    if state.pending_answer is not None:

        st.warning(
            "上一条回答已保存，但分析失败。"
            "无需重新输入原文。"
        )

        if st.button(
            "重新分析暂存回答",
            key="retry_verification",
        ):

            send(
                "/retryverify",
                graph,
            )

            st.rerun()

    elif session.stage == "awaiting_verification":

        st.info(
            "当前问题："
            + state.pending_question
        )

    if state.completed:

        st.success(
            "本次验证已完成。"
            "结论仍然属于待核验的用户自述证据。"
        )

    if session.stage == "awaiting_verification":

        if st.button(
            "取消当前验证",
            key="cancel_verify",
        ):

            send(
                "/cancelverify",
                graph,
            )

            st.rerun()


# ============================================
# 19. 主页面
# ============================================

def main():

    init_state()

    memory, graph = get_services()

    # ========================================
    # 20. 左侧侧边栏
    # ========================================

    with st.sidebar:

        st.title(
            "GrowthMentor-Agent"
        )

        st.caption(
            "本地个人成长导师控制台"
        )

        st.text_input(
            "会话标识",
            key="session_id_input",
        )

        if st.button(
            "打开会话"
        ):

            try:
                name = validate_session_id(
                    st.session_state[
                        "session_id_input"
                    ]
                )

            except ValueError as exc:
                st.error(str(exc))

            else:
                st.session_state["session_id"] = name
                st.session_state["messages"] = []
                st.session_state["last_path"] = []
                st.session_state["last_timeline"] = []
                st.session_state["pending_command"] = None
                st.session_state["guidance_hint"] = ""

                st.rerun()

        st.write(
            "当前会话：",
            st.session_state["session_id"],
        )

        st.divider()

        # 快捷业务操作
        quick_actions = [
            ("开始优势探索", "/start"),
            ("跨经历分析", "/compare"),
            ("查看会话状态", "/status"),
            ("保存会话", "/save"),
            ("放弃未完成任务", "/reset"),
        ]

        for label, command in quick_actions:

            if st.button(
                label,
                key=f"quick_{command}",
            ):

                send(command, graph)
                st.rerun()

        st.divider()

        # 导出原始Graph结构
        mermaid = (
            graph.get_graph().draw_mermaid()
        )

        st.download_button(
            "导出 LangGraph Mermaid",
            data=mermaid,
            file_name="growthmentor_graph.mmd",
            mime="text/plain",
        )

        st.caption(
            "当前为本地开发版本，"
            "尚未加入用户身份认证和数据库加密。"
        )

    # ========================================
    # 21. 读取最新持久化业务状态
    # ========================================

    session = memory.load_session(
        st.session_state["session_id"]
    )

    summary = get_summary(session)

    st.title(
        "个人成长导师 · Agent 控制台"
    )

    st.caption(
        "LangGraph + RAG + Assessment "
        "+ Memory + Dynamic Verification"
    )

    metrics = st.columns(4)

    titles = [
        "历史经历",
        "单经历分析",
        "候选优势方向",
        "已完成验证轮数",
    ]

    values = [
        summary["experiences"],
        summary["assessments"],
        summary["candidates"],
        summary["verification_rounds"],
    ]

    for column, title, value in zip(
        metrics,
        titles,
        values,
    ):
        column.metric(title, value)

    st.divider()

    # ========================================
    # 22. 上半部分：聊天 + 节点图
    # ========================================

    chat_col, graph_col = st.columns(
        [1.2, 1],
        gap="large",
    )

    with chat_col:

        st.subheader(
            "聊天与引导"
        )

        st.info(
            stage_description(session)
        )

        show_messages()
        show_guidance(session)

        question = st.chat_input(
            "输入问题或当前阶段的真实回答"
        )

        if question:

            send(
                question,
                graph,
            )

            st.rerun()

    # ========================================
    # 23. 创建右侧更新占位容器
    # ========================================

    with graph_col:

        st.subheader(
            "LangGraph 节点图 · 逐节点更新"
        )

        path = st.session_state["last_path"]

        graph_slot = st.empty()

        graph_slot.graphviz_chart(
            build_graphviz_dot(path),
            width="stretch",
        )

        st.caption(
            "绿色表示已确认完成的节点；"
            "灰色表示本轮未经过的节点。"
        )

        st.markdown(
            "**当前 / 最近执行路径**"
        )

        path_slot = st.empty()

        path_slot.code(
            format_trace(path),
            language="text",
        )

        status_slot = st.empty()
        timeline_slot = st.empty()

        if st.session_state["last_timeline"]:

            timeline_slot.code(
                "\n".join(
                    st.session_state["last_timeline"]
                ),
                language="text",
            )

        with st.expander(
            "LangGraph原始Mermaid"
        ):

            st.code(
                mermaid,
                language="mermaid",
            )

    # ========================================
    # 24. 下半部分：持久化用户记录
    # ========================================

    st.divider()

    st.subheader(
        "个人成长记录"
    )

    tab1, tab2, tab3 = st.tabs([
        "历史经历",
        "跨经历假设",
        "动态验证记录",
    ])

    with tab1:
        show_experiences(
            session,
            graph,
        )

    with tab2:
        show_cross_report(
            session,
            graph,
        )

    with tab3:
        show_verification(
            session,
            graph,
        )

    # ========================================
    # 25. 页面构造完毕后执行流式任务
    # ========================================

    run_pending_command(
        graph=graph,
        graph_slot=graph_slot,
        path_slot=path_slot,
        status_slot=status_slot,
        timeline_slot=timeline_slot,
    )


if __name__ == "__main__":
    main()
