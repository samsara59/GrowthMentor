
"""第75课：Supervisor任务规划与工作流执行页面。"""

from pathlib import Path

import streamlit as st

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)

from langgraph.types import Command

from app.agent.adaptive_mentor import (
    OllamaJsonModel,
)

from app.agent.experiment_protocol import (
    ExperimentDesigner,
)

from app.agent.supervisor_graph import (
    SupervisorPlanner,
    CoachingSynthesizer,
    build_supervisor_graph,
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


ROOT = Path(__file__).resolve().parents[3]

DB = (
    ROOT / "data"
    / "supervisor75_checkpoints.sqlite"
)

USER_DB = (
    ROOT / "data"
    / "user_memory.db"
)


st.set_page_config(
    page_title="Supervisor任务调度",
    layout="wide",
)


# ============================================
# 1. 使用已有Gemma 3和工具
# ============================================

@st.cache_resource(
    show_spinner=False
)
def services():

    registry = build_capability_registry()

    source_status = connect_source_search(
        registry,
        ROOT,
    )

    model = OllamaJsonModel(
        model="gemma3:4b"
    )

    return (
        SupervisorPlanner(model),
        registry,
        ExperimentDesigner(model),
        CoachingSynthesizer(model),
        source_status,
    )


# ============================================
# 2. 将节点事件实时显示到网页
# ============================================

def stream(
    graph,
    inp,
    cfg,
):

    with st.status(
        "LangGraph 正在执行",
        expanded=True,
    ) as status:

        for chunk in graph.stream(
            inp,
            config=cfg,
            stream_mode="updates",
        ):

            for node, update in chunk.items():

                if node == "__interrupt__":

                    st.write(
                        "已暂停：等待审批"
                    )

                else:

                    st.write(
                        "节点完成：",
                        node,
                    )

                    if (
                        isinstance(update, dict)
                        and update.get("phase")
                    ):

                        st.caption(
                            "执行状态："
                            + update["phase"]
                        )

        status.update(
            label="本次执行已暂停或完成",
            state="complete",
        )


# ============================================
# 3. 页面主函数
# ============================================

def main():

    st.title(
        "第75课 · Supervisor任务分解"
    )

    st.caption(
        "LLM制定任务计划 → "
        "用户审批 → "
        "专门Worker执行 → "
        "Coaching整合"
    )

    consent = st.checkbox(
        "同意本次工作流保存问题、计划、"
        "任务结果和回复到本地SQLite",
        value=False,
    )

    if not consent:

        st.info(
            "本页包含可持久保存的个人问题。"
            "请先授权本地检查点。"
        )

        return

    sid_input = st.text_input(
        "会话ID",
        value="lesson75_demo",
    )

    try:

        sid = check_session_id(
            sid_input
        )

    except ValueError as exc:

        st.error(str(exc))
        return

    thread_id = (
        "lg75_" + sid
    )

    cfg = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    try:

        (
            planner,
            registry,
            designer,
            coaching,
            source_status,
        ) = services()

    except Exception as exc:

        st.error(
            "初始化失败："
            f"{type(exc).__name__}: {exc}"
        )

        return

    st.caption(
        "知识检索状态："
        + str(source_status)
    )

    DB.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================
    # 4. 使用独立的SQLite检查点
    # ========================================

    with SqliteSaver.from_conn_string(
        str(DB)
    ) as saver:

        graph = build_supervisor_graph(
            planner,
            registry,
            designer,
            coaching,
            saver,
        )

        snap = graph.get_state(
            cfg
        )

        values = snap.values or {}

        pending = any(
            getattr(
                task,
                "interrupts",
                (),
            )
            for task in snap.tasks
        )

        st.write(
            "线程ID：",
            thread_id,
        )

        st.write(
            "工作流状态：",
            values.get(
                "phase",
                "未启动",
            ),
        )

        st.write(
            "待执行节点：",
            list(snap.next or ()),
        )

        # ====================================
        # 5. 启动新的任务规划
        # ====================================

        if not snap.next:

            with st.form(
                "new_task"
            ):

                goal = st.text_input(
                    "成长目标",

                    value=(
                        "找到愿意长期投入的方向"
                    ),
                )

                question = st.text_area(
                    "当前问题",

                    value=(
                        "我想用一周的小项目"
                        "比较算法研究和产品开发，"
                        "请给出可执行计划。"
                    ),
                )

                use_notes = st.checkbox(
                    "额外读取并保存到本轮检查点："
                    "已确认的长期笔记",
                    value=False,
                )

                start = (
                    st.form_submit_button(
                        "让Supervisor制定任务计划"
                    )
                )

            if start:

                try:

                    notes = []

                    if use_notes:

                        mem = GrowthMemoryStore(
                            USER_DB
                        )

                        notes = [
                            {
                                "category": n[
                                    "category"
                                ],

                                "text": n[
                                    "content"
                                ],
                            }

                            for n in mem.list_notes(
                                sid,
                                limit=8,
                            )
                        ]

                    stream(
                        graph,

                        {
                            "goal": goal,
                            "question": question,

                            "confirmed_notes": notes,
                        },

                        cfg,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "规划失败："
                        f"{type(exc).__name__}: {exc}"
                    )

        # ====================================
        # 6. Supervisor计划等待用户审批
        # ====================================

        elif pending:

            st.subheader(
                "Supervisor制定的任务计划"
            )

            plan = values.get(
                "plan"
            ) or {}

            st.write(
                "规划说明：",
                plan.get(
                    "rationale",
                    "",
                ),
            )

            if not plan.get(
                "tasks"
            ):

                st.write(
                    "无需前置任务，"
                    "审批后直接调用Coaching Worker。"
                )

            for item in plan.get(
                "tasks",
                [],
            ):

                with st.container(
                    border=True
                ):

                    st.write(
                        f"{item['task_id']} · "
                        f"{item['worker']}"
                    )

                    st.write(
                        item["objective"]
                    )

                    if item.get(
                        "query"
                    ):

                        st.caption(
                            "检索："
                            + item["query"]
                        )

                    if item.get(
                        "comparison_options"
                    ):

                        st.caption(
                            "比较："
                            + " / ".join(
                                item[
                                    "comparison_options"
                                ]
                            )
                        )

            st.warning(
                "批准的是上述计划的执行。"
                "规划阶段已经调用Gemma 3；"
                "拒绝后不会执行专门Worker。"
            )

            a, b = st.columns(2)

            if a.button(
                "批准并执行任务",
                type="primary",
            ):

                try:

                    stream(
                        graph,
                        Command(resume=True),
                        cfg,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "执行失败，可使用同线程"
                        "检查点恢复："
                        f"{type(exc).__name__}: {exc}"
                    )

            if b.button(
                "拒绝此计划"
            ):

                try:

                    stream(
                        graph,
                        Command(resume=False),
                        cfg,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "取消失败："
                        f"{type(exc).__name__}: {exc}"
                    )

        # ====================================
        # 7. 故障后重试未完成的节点
        # ====================================

        else:

            st.warning(
                "检测到未完成且非审批中断的节点，"
                "可能是工作节点失败。"
            )

            if st.button(
                "从最近检查点重试"
            ):

                try:

                    stream(
                        graph,
                        None,
                        cfg,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "重试失败："
                        f"{type(exc).__name__}: {exc}"
                    )

        # ====================================
        # 8. 显示保存的任务协议
        # ====================================

        if values.get(
            "plan"
        ):

            with st.expander(
                "保存的任务协议"
            ):

                st.json(
                    values["plan"]
                )

        # ====================================
        # 9. 显示Worker实际交付物
        # ====================================

        if values.get(
            "worker_results"
        ):

            st.subheader(
                "Worker实际交付结果"
            )

            for result in values[
                "worker_results"
            ]:

                with st.expander(
                    f"{result['task_id']} · "
                    f"{result['worker']} · "
                    f"{result['status']}",

                    expanded=True,
                ):

                    if (
                        result["worker"]
                        == "knowledge"
                    ):

                        for item in result.get(
                            "evidence",
                            [],
                        ):

                            st.write(
                                f"[{item['evidence_id']}] "
                                f"{item['source']} · "
                                f"{item['location']}"
                            )

                            st.caption(
                                item["quote"]
                            )

                        st.caption(
                            "这些是检索候选，"
                            "尚未逐字核验。"
                        )

                    elif (
                        result["worker"]
                        == "practice"
                    ):

                        if result.get(
                            "preview"
                        ):

                            st.markdown(
                                result["preview"]
                            )

                        st.caption(
                            "方案尚未写入PracticeStore，"
                            "也不代表用户已经执行。"
                        )

                    if result.get(
                        "error"
                    ):

                        st.error(
                            result["error"]
                        )

        # ====================================
        # 10. 显示Coaching最终回答
        # ====================================

        if values.get(
            "answer"
        ):

            st.subheader(
                "最终指导"
            )

            st.write(
                values["answer"]
            )

            if values.get(
                "evidence_ids"
            ):

                st.caption(
                    "使用的教材候选编号："
                    + ", ".join(
                        values[
                            "evidence_ids"
                        ]
                    )
                )

            if values.get(
                "limitations"
            ):

                st.write(
                    "局限性："
                    + values[
                        "limitations"
                    ]
                )

        # ====================================
        # 11. 查看检查点历史
        # ====================================

        with st.expander(
            "最近10个检查点"
        ):

            for item in saver.list(
                cfg,
                limit=10,
            ):

                st.caption(
                    str(
                        item.config.get(
                            "configurable",
                            {},
                        ).get(
                            "checkpoint_id"
                        )
                    )
                )

        st.caption(
            "本页是第75课实验入口，"
            "尚无登录鉴权，不应公开部署。"
        )


if __name__ == "__main__":
    main()
