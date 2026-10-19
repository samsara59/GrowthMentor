
"""第74课：LangGraph节点事件、持久化中断与恢复。"""

import os
from pathlib import Path

import streamlit as st

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)
from langgraph.types import Command

from app.agent.adaptive_mentor import (
    AdaptiveMentor,
    MentorPlanner,
    OllamaJsonModel,
)
from app.agent.growth_graph import (
    build_growth_graph,
)
from app.capabilities.registry import (
    build_capability_registry,
)
from app.capabilities.source_search import (
    connect_source_search,
)
from app.memory.growth_store import (
    GrowthMemoryStore,
    check_session_id,
)
from app.methods.catalog import (
    build_method_catalog,
)
from app.programs.catalog import (
    build_program_catalog,
)


ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_DB = (
    ROOT / "data" / "workflow74_checkpoints.sqlite"
)

USER_DB = (
    ROOT / "data" / "user_memory.db"
)

st.set_page_config(
    page_title="LangGraph工作流恢复",
    layout="wide",
)


# ============================================
# 1. 复用现有导师和工具
# ============================================

@st.cache_resource(show_spinner=False)
def get_mentor():

    registry = build_capability_registry()

    source_status = connect_source_search(
        registry,
        ROOT,
    )

    mentor = AdaptiveMentor(
        MentorPlanner(
            OllamaJsonModel(
                model="gemma3:4b"
            )
        ),
        registry,
        build_method_catalog(),
    )

    return mentor, source_status


# ============================================
# 2. Stream事件实际输出到网页
# ============================================

def run_with_stream(
    graph,
    payload,
    config,
    event_key: str,
):

    events = []

    with st.status(
        "LangGraph 正在执行",
        expanded=True,
    ) as status:

        # updates：节点完成后的状态更新
        # custom：节点内部的自定义进度
        for mode, data in graph.stream(
            payload,
            config=config,
            stream_mode=[
                "updates",
                "custom",
            ],
        ):

            if mode == "custom":

                message = str(
                    data.get(
                        "message",
                        "自定义事件",
                    )
                )

                events.append(message)

                st.write(
                    "进度：",
                    message,
                )

            elif mode == "updates":

                for node, updated in data.items():

                    if node == "__interrupt__":

                        message = (
                            "工作流已暂停，"
                            "等待用户确认"
                        )

                    else:

                        phase = (
                            updated.get(
                                "phase", ""
                            )
                            if isinstance(
                                updated, dict
                            )
                            else ""
                        )

                        message = (
                            f"节点 {node} 完成"
                            + (
                                f"；状态={phase}"
                                if phase
                                else ""
                            )
                        )

                    events.append(
                        message
                    )

                    st.write(
                        message
                    )

        status.update(
            label="本次运行已暂停或结束",
            state="complete",
        )

    st.session_state[
        event_key
    ] = events


# ============================================
# 3. 主界面
# ============================================

def main():

    st.title(
        "LangGraph · 实时执行与工作流恢复"
    )

    st.caption(
        "复用现有成长导师。"
        "通过节点事件观察执行，"
        "通过SQLite检查点恢复中断状态。"
    )

    consent = st.checkbox(
        "我同意将本页面的对话和执行状态"
        "保存到本机检查点数据库",
        value=False,
    )

    if not consent:

        st.info(
            "请先确认是否启用本地检查点。"
            "此页面的工作流状态会包含"
            "用户问题和导师回复。"
        )

        return

    # ========================================
    # 4. 选择工作流线程
    # ========================================

    sid_text = st.text_input(
        "本地会话ID",
        value="my_growth",
    )

    try:

        sid = check_session_id(
            sid_text
        )

    except ValueError as exc:

        st.error(str(exc))
        return

    # 与旧版成长记忆会话同名，
    # 但LangGraph检查点使用独立命名空间。
    thread_id = "lg74_" + sid

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    event_key = (
        "langgraph74_events_"
        + thread_id
    )

    programs = {
        p.program_id: p
        for p in build_program_catalog().all()
    }

    program_id = st.selectbox(
        "当前Growth Program",
        options=list(programs),
        format_func=lambda k: (
            programs[k].name
        ),
    )

    goal = st.text_area(
        "成长目标",

        value=(
            programs[program_id].objective
        ),

        key="lg74_goal_" + program_id,
    )

    include_notes = st.checkbox(
        "本轮额外读取该会话已确认的长期笔记",
        value=False,
    )

    mentor, source_status = get_mentor()

    st.caption(
        "教材检索：" + source_status
    )

    # ========================================
    # 5. 使用独立SQLite Checkpointer
    # ========================================

    CHECKPOINT_DB.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 限制检查点序列化可加载的模块类型
    os.environ.setdefault(
        "LANGGRAPH_STRICT_MSGPACK",
        "true",
    )

    # with保证每次Streamlit脚本结束后
    # SQLite连接会正常关闭。
    #
    # 下一次页面重载重新打开相同文件，
    # 不会丢失已提交的检查点。
    with SqliteSaver.from_conn_string(
        str(CHECKPOINT_DB)
    ) as checkpointer:

        graph = build_growth_graph(
            mentor,
            checkpointer,
        )

        snapshot = graph.get_state(
            config
        )

        values = snapshot.values or {}

        # ====================================
        # 6. 检查是否存在暂停节点
        # ====================================

        pending_interrupt = any(
            bool(
                getattr(
                    task,
                    "interrupts",
                    (),
                )
            )
            for task in snapshot.tasks
        )

        next_nodes = tuple(
            snapshot.next or ()
        )

        phase = values.get(
            "phase",
            "尚未运行",
        )

        a, b, c = st.columns(3)

        a.metric(
            "当前线程",
            thread_id,
        )

        b.metric(
            "执行状态",
            phase,
        )

        c.metric(
            "等待执行的节点数",
            len(next_nodes),
        )

        checkpoint_id = (
            snapshot.config or {}
        ).get(
            "configurable",
            {},
        ).get(
            "checkpoint_id",
        )

        if checkpoint_id:

            st.caption(
                "最近检查点ID："
                + str(checkpoint_id)
            )

        # ====================================
        # 7. 显示本轮已有对话
        # ====================================

        st.subheader(
            "已经保存的工作流对话"
        )

        for entry in values.get(
            "history",
            [],
        ):

            if not isinstance(
                entry,
                dict,
            ):
                continue

            role = entry.get(
                "role",
                "assistant",
            )

            if role not in (
                "user",
                "assistant",
            ):
                continue

            with st.chat_message(
                role
            ):
                st.write(
                    entry.get(
                        "content", ""
                    )
                )

        # ====================================
        # 8. 首次运行或上一轮已结束
        # ====================================

        if not next_nodes:

            st.subheader(
                "发起新一轮LangGraph任务"
            )

            with st.form(
                "lg74_start"
            ):

                question = st.text_area(
                    "本轮成长问题",

                    value=(
                        "我能够解决复杂的机器学习问题，"
                        "但是如何区分自然擅长"
                        "和真正愿意长期投入？"
                    ),

                    max_chars=1500,
                )

                start = (
                    st.form_submit_button(
                        "提交问题并停在确认节点"
                    )
                )

            if start:

                try:

                    notes = []

                    if include_notes:

                        memory = GrowthMemoryStore(
                            USER_DB
                        )

                        notes = [
                            {
                                "category": n["category"],
                                "text": n["content"],
                            }
                            for n in memory.list_notes(
                                sid,
                                limit=12,
                            )
                        ]

                    run_with_stream(
                        graph,

                        {
                            "question": question.strip(),
                            "goal": goal.strip(),
                            "program_id": program_id,
                            "confirmed_notes": notes,
                        },

                        config,
                        event_key,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "启动失败："
                        f"{type(exc).__name__}: {exc}"
                    )

        # ====================================
        # 9. 从持久化中断点恢复
        # ====================================

        elif pending_interrupt:

            st.subheader(
                "检测到等待用户确认的工作流"
            )

            st.write(
                values.get(
                    "preview",
                    values.get(
                        "question", ""
                    ),
                )
            )

            st.info(
                "此状态已记录在SQLite中。"
                "现在可以关闭Streamlit，"
                "重新启动后使用相同会话ID恢复。"
            )

            left, right = st.columns(2)

            if left.button(
                "确认：继续调用Gemma 3",
                type="primary",
            ):

                try:

                    run_with_stream(
                        graph,
                        Command(
                            resume=True
                        ),
                        config,
                        event_key,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "恢复失败："
                        f"{type(exc).__name__}: {exc}"
                    )

            if right.button(
                "拒绝：取消本轮执行"
            ):

                try:

                    run_with_stream(
                        graph,
                        Command(
                            resume=False
                        ),
                        config,
                        event_key,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "取消失败："
                        f"{type(exc).__name__}: {exc}"
                    )

        # ====================================
        # 10. 节点错误后从检查点重试
        # ====================================

        else:

            st.warning(
                "工作流存在尚未完成的节点，"
                "但不是等待人工确认的中断。"
                "可能是此前执行出现了错误。"
            )

            st.write(
                "待执行节点：",
                next_nodes,
            )

            if st.button(
                "从最近检查点重试"
            ):

                try:

                    run_with_stream(
                        graph,
                        None,
                        config,
                        event_key,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "重试失败："
                        f"{type(exc).__name__}: {exc}"
                    )

        # ====================================
        # 11. 显示检查点中的最终结果
        # ====================================

        if values.get("result"):

            st.subheader(
                "最近一次运行结果"
            )

            st.write(
                values["result"]
            )

            st.caption(
                "Agent动作："
                + str(
                    values.get(
                        "action", "未记录"
                    )
                )
            )

            if values.get(
                "trace"
            ):

                with st.expander(
                    "原有AdaptiveMentor内部行动记录"
                ):

                    for step in values["trace"]:
                        st.write(step)

        # ====================================
        # 12. 最近一次实时事件
        # ====================================

        with st.expander(
            "最近一次页面接收到的节点事件",
            expanded=True,
        ):

            events = (
                st.session_state.get(
                    event_key,
                    [],
                )
            )

            if events:

                for i, event in enumerate(
                    events,
                    1,
                ):

                    st.write(
                        f"{i}. {event}"
                    )

            else:

                st.caption(
                    "新运行后显示节点事件。"
                    "浏览器会话丢失时，"
                    "这里的临时事件列表"
                    "可能为空，但SQLite检查点仍在。"
                )

        # ====================================
        # 13. 本线程的检查点历史
        # ====================================

        with st.expander(
            "查看已保存的检查点历史"
        ):

            history = list(
                checkpointer.list(
                    config,
                    limit=10,
                )
            )

            st.write(
                "最近检查点数量：",
                len(history),
            )

            for item in history:

                cp_id = (
                    item.config.get(
                        "configurable",
                        {},
                    ).get(
                        "checkpoint_id"
                    )
                )

                st.caption(
                    str(cp_id)
                )

        # ====================================
        # 14. 删除本线程的检查点
        # ====================================

        st.divider()

        delete_agreed = st.checkbox(
            "我确认删除本线程全部LangGraph检查点"
        )

        if st.button(
            "删除本线程检查点",
            disabled=not delete_agreed,
        ):

            try:

                checkpointer.delete_thread(
                    thread_id
                )

                st.session_state.pop(
                    event_key,
                    None,
                )

                st.rerun()

            except Exception as exc:

                st.error(
                    "删除失败："
                    f"{type(exc).__name__}: {exc}"
                )

    st.caption(
        "本课使用独立的workflow74_checkpoints.sqlite。"
        "不向旧用户记忆数据库保存图对话。"
        "检查点目前未加密，不应公开部署。"
    )


if __name__ == "__main__":
    main()
