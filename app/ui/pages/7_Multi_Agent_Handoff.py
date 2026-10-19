
"""第76课：可切换场景、独立线程、可恢复的Multi-Agent实验页面。"""

from pathlib import Path

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.agent.adaptive_mentor import OllamaJsonModel
from app.agent.experiment_protocol import ExperimentDesigner
from app.agent.supervisor_graph import (
    SupervisorPlanner,
    CoachingSynthesizer,
)
from app.agent.multi_agent_handoff import build_multi_agent_graph
from app.capabilities.registry import build_capability_registry
from app.capabilities.source_search import connect_source_search
from app.knowledge.source_provenance import SourceCatalog
from app.memory.growth_store import GrowthMemoryStore, check_session_id


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DB = ROOT / "data" / "multiagent76_checkpoints.sqlite"
USER_DB = ROOT / "data" / "user_memory.db"


# ============================================
# 1. 场景配置：只预设输入，不固定Supervisor决策
# ============================================

SCENARIOS = {
    "s1": {
        "title": "场景1：知识检索与原文核验",
        "goal": "结合可靠教材理解自发兴趣与自然擅长的区别",
        "question": (
            "我能解决复杂的机器学习问题，但这是否意味着我真正喜欢科研？"
            "请结合实际教材，区分擅长完成任务与自发投入。"
        ),
    },

    "s2": {
        "title": "场景2：A/B实践方案设计",
        "goal": "通过短期实践比较两种成长方向的真实体验",
        "question": (
            "我想在一周内比较算法研究和产品开发。"
            "请设计两个基于同一问题、时间预算相近的实践任务，"
            "分别观察主动投入意愿、专注感和受阻反应。"
        ),
    },

    "s3": {
        "title": "场景3：一个Knowledge Worker执行两项任务",
        "goal": "查找自发兴趣与自然擅长行为方式的不同证据",
        "question": (
            "请分别查找教材中关于自发兴趣、自然擅长行为方式的内容，"
            "作为两个独立的检索任务，再解释两者的区别。"
            "不要直接推断我的优势。"
        ),
    },

    "custom": {
        "title": "自定义成长问题",
        "goal": "根据用户当前成长目标提供有依据的指导",
        "question": "我想讨论一个具体的成长问题。",
    },

    "legacy": {
        "title": "查看旧版未分场景的线程",
        "goal": "",
        "question": "",
    },
}


st.set_page_config(
    page_title="第76课 · 多Agent场景实验",
    layout="wide",
)


# ============================================
# 2. 复用现有Gemma 3、检索与实践设计器
# ============================================

@st.cache_resource(show_spinner=False)
def services():

    registry = build_capability_registry()

    source_status = connect_source_search(
        registry,
        ROOT,
    )

    model = OllamaJsonModel(model="gemma3:4b")

    catalog = SourceCatalog(ROOT / "books")

    return (
        SupervisorPlanner(model),
        registry,
        ExperimentDesigner(model),
        CoachingSynthesizer(model),
        catalog,
        source_status,
    )


# ============================================
# 3. 显示真实的LangGraph节点事件
# ============================================

def run_graph(graph, payload, config):

    with st.status(
        "LangGraph 正在执行",
        expanded=True,
    ) as panel:

        for chunk in graph.stream(
            payload,
            config=config,
            stream_mode="updates",
        ):

            for node, update in chunk.items():

                if node == "__interrupt__":

                    st.write(
                        "暂停：等待用户审批Supervisor任务计划"
                    )

                else:

                    st.write(f"节点完成：{node}")

                    if (
                        isinstance(update, dict)
                        and update.get("phase")
                    ):

                        st.caption(
                            f"当前阶段：{update['phase']}"
                        )

        panel.update(
            label="本次执行已暂停或完成",
            state="complete",
        )


# ============================================
# 4. Streamlit主页面
# ============================================

def main():

    st.title(
        "第76课 · 可切换场景的Multi-Agent实验"
    )

    st.caption(
        "各场景使用独立的LangGraph线程；"
        "切换场景不会删除另一场景的检查点。"
    )

    consent = st.checkbox(
        "同意将实验问题、Agent交接消息及回答保存在本机SQLite",
        value=False,
    )

    if not consent:

        st.info(
            "勾选后才会读取或写入本地工作流检查点。"
        )

        return

    # ========================================
    # 5. 会话ID与场景选择
    # ========================================

    sid_input = st.text_input(
        "用户会话ID",
        value="lesson76_demo",
    )

    try:

        sid = check_session_id(
            sid_input
        )

    except ValueError as exc:

        st.error(str(exc))
        return

    scene_id = st.selectbox(
        "选择要运行的场景",
        options=list(SCENARIOS),
        format_func=lambda key: (
            SCENARIOS[key]["title"]
        ),
    )

    # ========================================
    # 6. 每个场景拥有自己的thread_id
    # ========================================

    if scene_id == "legacy":

        # 兼容第76课原版的旧线程
        thread_id = "lg76_" + sid

        st.caption(
            "兼容旧版线程："
            "可查看此前暂停的任务并继续审批。"
        )

    else:

        round_key = (
            f"lesson76_run_{sid}_{scene_id}"
        )

        a, b = st.columns([3, 1])

        with a:

            run_number = st.number_input(
                "本场景实验轮次（同一轮次可在重启后恢复）",
                min_value=1,
                max_value=999,
                step=1,
                key=round_key,
            )

        with b:

            st.write(" ")

            if st.button(
                "新开一轮",
                key=f"new_round_{sid}_{scene_id}",
            ):

                if int(
                    st.session_state[round_key]
                ) >= 999:

                    st.error(
                        "轮次已达到上限，"
                        "请使用新的用户会话ID。"
                    )

                else:

                    st.session_state[round_key] += 1

                    st.rerun()

        thread_id = (
            f"lg76_{sid}_{scene_id}_r{int(run_number)}"
        )

    st.write("本场景的实际 LangGraph 线程：")

    st.code(
        thread_id,
        language="text",
    )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # ========================================
    # 7. 加载服务
    # ========================================

    try:

        (
            planner,
            registry,
            designer,
            coaching,
            catalog,
            source_status,
        ) = services()

    except Exception as exc:

        st.error(
            "服务初始化失败："
            f"{type(exc).__name__}: {exc}"
        )

        return

    st.caption(
        "教材检索状态：" + str(source_status)
    )

    CHECKPOINT_DB.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================
    # 8. SQLite检查点
    # ========================================

    with SqliteSaver.from_conn_string(
        str(CHECKPOINT_DB)
    ) as saver:

        graph = build_multi_agent_graph(
            planner,
            registry,
            designer,
            coaching,
            catalog,
            saver,
        )

        snap = graph.get_state(
            config
        )

        state = snap.values or {}

        next_nodes = tuple(
            snap.next or ()
        )

        waiting = any(
            bool(
                getattr(
                    task,
                    "interrupts",
                    (),
                )
            )
            for task in snap.tasks
        )

        x, y = st.columns(2)

        x.metric(
            "执行状态",
            state.get(
                "phase",
                "未启动",
            ),
        )

        y.metric(
            "待执行节点数",
            len(next_nodes),
        )

        # ====================================
        # 9. 新场景：允许启动新任务
        # ====================================

        if not next_nodes and not state:

            if scene_id == "legacy":

                st.warning(
                    "旧版线程中没有任务。"
                    "请切换到场景1、2、3或自定义。"
                )

            else:

                scene = SCENARIOS[scene_id]

                with st.form(
                    f"start_{thread_id}"
                ):

                    goal = st.text_input(
                        "成长目标",
                        value=scene["goal"],
                        key=f"goal_{thread_id}",
                    )

                    question = st.text_area(
                        "成长问题（可以修改）",
                        value=scene["question"],
                        key=f"question_{thread_id}",
                        height=135,
                    )

                    load_notes = st.checkbox(
                        "本轮额外读取此会话已确认的长期笔记",
                        value=False,
                    )

                    start = st.form_submit_button(
                        "生成本场景Supervisor计划"
                    )

                if start:

                    try:

                        notes = []

                        if load_notes:

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
                                    limit=8,
                                )
                            ]

                        run_graph(
                            graph,
                            {
                                "goal": goal.strip(),
                                "question": question.strip(),
                                "confirmed_notes": notes,
                            },
                            config,
                        )

                        st.rerun()

                    except Exception as exc:

                        st.error(
                            "规划失败："
                            f"{type(exc).__name__}: {exc}"
                        )

        # ====================================
        # 10. 已暂停：允许批准、拒绝或切换场景
        # ====================================

        elif waiting:

            st.subheader(
                "Supervisor任务计划 · 等待审批"
            )

            st.json(
                state.get("plan", {})
            )

            st.info(
                "此场景正在等待审批。"
                "你可以切换到场景2，"
                "并在以后回到本场景继续。"
            )

            left, right = st.columns(2)

            if left.button(
                "批准并执行",
                key=f"approve_{thread_id}",
                type="primary",
            ):

                try:

                    run_graph(
                        graph,
                        Command(resume=True),
                        config,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "执行失败："
                        f"{type(exc).__name__}: {exc}"
                    )

            if right.button(
                "拒绝本轮计划",
                key=f"reject_{thread_id}",
            ):

                try:

                    run_graph(
                        graph,
                        Command(resume=False),
                        config,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "取消失败："
                        f"{type(exc).__name__}: {exc}"
                    )

        # ====================================
        # 11. 中途故障：继续重试此线程
        # ====================================

        elif next_nodes:

            st.warning(
                "当前工作流处于未完成的非审批节点，"
                "可从检查点重试。"
            )

            st.write(
                "待执行节点：",
                next_nodes,
            )

            if st.button(
                "从当前检查点重试",
                key=f"retry_{thread_id}",
            ):

                try:

                    run_graph(
                        graph,
                        None,
                        config,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "重试失败："
                        f"{type(exc).__name__}: {exc}"
                    )

        # ====================================
        # 12. 本轮完成：保留历史，允许新开一轮
        # ====================================

        else:

            st.success(
                "本轮工作流已经结束。"
                "如需同一场景的新任务，"
                "点击上面的‘新开一轮’；"
                "若要换场景，直接切换选择框。"
            )

        # ====================================
        # 13. 显示任务计划
        # ====================================

        if state.get("plan"):

            with st.expander(
                "Supervisor保存的任务计划"
            ):

                st.json(
                    state["plan"]
                )

        # ====================================
        # 14. 展示实际Agent交接消息
        # ====================================

        if state.get("handoffs"):

            st.subheader(
                "真实Agent交接消息"
            )

            for i, msg in enumerate(
                state["handoffs"],
                1,
            ):

                label = (
                    f"{i}. {msg['task_id']} · "
                    f"{msg['sender']} → "
                    f"{msg['recipient']} · "
                    f"{msg['status']} · "
                    f"第{msg['attempt']}次"
                )

                with st.expander(label):

                    st.json(msg)

        # ====================================
        # 15. 展示Worker交付物
        # ====================================

        if state.get("worker_results"):

            st.subheader(
                "Worker实际交付物"
            )

            for result in state["worker_results"]:

                with st.expander(
                    f"{result['task_id']} · "
                    f"{result['worker']} · "
                    f"{result['status']}",
                    expanded=True,
                ):

                    if result["worker"] == "knowledge":

                        for hit in result.get(
                            "evidence",
                            [],
                        ):

                            st.markdown(
                                f"**[{hit['evidence_id']}] "
                                f"{hit['source']}**"
                            )

                            st.caption(
                                f"位置："
                                f"{hit.get('location', '未知')}；"
                                f"原文校验："
                                f"{hit.get('verification', '未知')}"
                            )

                            st.write(
                                hit.get("quote", "")
                            )

                    else:

                        if result.get("preview"):

                            st.markdown(
                                result["preview"]
                            )

                    if result.get("error"):

                        st.error(
                            str(result["error"])
                        )

        # ====================================
        # 16. 显示Coaching最终回答
        # ====================================

        if state.get("answer"):

            st.subheader(
                "Coaching最终回复"
            )

            st.write(
                state["answer"]
            )

            if state.get("evidence_ids"):

                st.caption(
                    "证据编号："
                    + ", ".join(
                        state["evidence_ids"]
                    )
                )

            if state.get("limitations"):

                st.write(
                    "限制："
                    + state["limitations"]
                )

        # ====================================
        # 17. 展示当前线程的检查点历史
        # ====================================

        with st.expander(
            "最近5个持久化检查点"
        ):

            for checkpoint in saver.list(
                config,
                limit=5,
            ):

                checkpoint_id = (
                    checkpoint.config.get(
                        "configurable",
                        {},
                    ).get("checkpoint_id")
                )

                st.caption(
                    str(checkpoint_id)
                )

    st.caption(
        "不同场景在同一SQLite文件内按thread_id隔离；"
        "本地实验无登录鉴权，不要部署到公网。"
    )


if __name__ == "__main__":
    main()
