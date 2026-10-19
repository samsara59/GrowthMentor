from app.agent.response_validation import GroundedResponseValidator
from app.core.ollama_nli_adapter import OllamaNLIAdapter

"""第68课：成长对话、实践确认、反馈及回顾。"""

import os
from pathlib import Path

import streamlit as st

from app.agent.adaptive_mentor import (
    AdaptiveMentor,
    MentorPlanner,
    MentorState,
    MentorTurn,
    OllamaJsonModel,
    ChatEntry,
)

from app.agent.dialogue_strategy import (
    build_growth_view,
    FOCUS_LABELS,
)

from app.agent.experiment_protocol import (
    ComparisonProtocol,
    render_protocol,
)

from app.capabilities.registry import (
    build_capability_registry,
)

from app.capabilities.source_search import (
    connect_source_search,
)

from app.domain.contracts import GrowthTask

from app.memory.growth_store import (
    CATEGORIES,
    GrowthMemoryStore,
    MemoryConflictError,
    check_session_id,
)

from app.memory.practice_store import PracticeStore

from app.core.model_provider import provider_from_env
from app.core.mentor_model_adapter import MentorModelAdapter

from app.methods.catalog import build_method_catalog
from app.programs.catalog import build_program_catalog


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "user_memory.db"

st.set_page_config(
    page_title="GrowthMentor-Agent",
    layout="wide",
)


# ============================================
# 1. 创建共享服务
# ============================================

@st.cache_resource(show_spinner=False)
def services():
    registry = build_capability_registry()

    source_status = connect_source_search(
        registry,
        ROOT,
    )

    api_url = os.getenv(
        "MODEL_BASE_URL", ""
    ).strip()

    api_name = os.getenv(
        "MODEL_NAME", ""
    ).strip()

    if bool(api_url) != bool(api_name):
        raise ValueError(
            "Configure both MODEL_BASE_URL and MODEL_NAME"
        )

    if api_url:
        provider = provider_from_env()
        model = MentorModelAdapter(provider)
        nli_provider = provider
    else:
        model = OllamaJsonModel(
            model=os.getenv(
                "GROWTH_OLLAMA_MODEL",
                "qwen3:4b",
            )
        )
        nli_provider = OllamaNLIAdapter(model)

    response_validator = GroundedResponseValidator(
        nli_provider
    )

    mentor = AdaptiveMentor(
        MentorPlanner(model),
        registry,
        build_method_catalog(),
        response_validator=response_validator,
    )

    return (
        mentor,
        GrowthMemoryStore(DB),
        PracticeStore(DB),
        source_status,
    )




# ============================================
# 2. 初始成长任务
# ============================================

def fresh(
    program_id: str,
    goal: str,
) -> MentorState:

    return MentorState(
        task=GrowthTask(
            task_id="mentor-" + program_id,
            program_id=program_id,
            goal=goal,
        )
    )


def initialize(
    program_id: str,
    goal: str,
):

    defaults = {
        "adaptive_state": fresh(
            program_id,
            goal,
        ),
        "adaptive_revision": 0,
        "adaptive_active_id": "my_growth",
        "adaptive_last": None,
        "adaptive_unsaved": False,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# ============================================
# 3. 保存一轮对话和生成的待确认方案
# ============================================

def save_turn(
    new_state,
    turn,
    *,
    save,
    sid,
    revision,
    memory,
    practices,
):

    if save:

        new_rev = memory.save_session(
            sid,
            new_state,
            revision,
        )

        st.session_state[
            "adaptive_revision"
        ] = new_rev

        # 只有结构化A/B方案才能登记
        if turn.experiment_protocol is not None:

            try:
                practices.create_proposal(
                    sid,
                    turn.experiment_protocol,
                )

            except Exception as exc:

                st.warning(
                    "对话已保存，"
                    "但实践方案未登记："
                    + str(exc)
                )

    st.session_state[
        "adaptive_state"
    ] = new_state

    st.session_state[
        "adaptive_last"
    ] = turn

    st.session_state[
        "adaptive_unsaved"
    ] = False


# ============================================
# 4. 主页面
# ============================================

def main():

    mentor, memory, practices, source_status = (
        services()
    )

    programs = {
        p.program_id: p
        for p in build_program_catalog().all()
    }

    st.title(
        "GrowthMentor-Agent · 长期成长导师"
    )

    st.caption(
        "自适应对话 → A/B实践 → 用户确认 "
        "→ 真实反馈 → LLM回顾"
    )

    # ========================================
    # 5. 左侧：成长目标和记忆
    # ========================================

    with st.sidebar:

        program_id = st.selectbox(
            "Growth Program",
            list(programs),
            format_func=lambda k: (
                programs[k].name
            ),
        )

        spec = programs[program_id]

        goal = st.text_area(
            "本次成长目标",
            value=spec.objective,
            key="goal_" + program_id,
        ).strip() or spec.objective

        st.caption(
            "任务目录状态："
            + str(spec.status)
        )

        st.divider()

        proposed_sid = st.text_input(
            "会话ID",
            value="my_growth",
        )

        save = st.checkbox(
            "同意将对话、方案和反馈"
            "保存在本机SQLite",
            value=False,
        )

        initialize(
            program_id,
            goal,
        )

        if st.button(
            "加载 / 新建此会话"
        ):

            try:

                sid = check_session_id(
                    proposed_sid
                )

                loaded, rev = (
                    memory.load_session(sid)
                    if save
                    else (None, 0)
                )

                st.session_state[
                    "adaptive_state"
                ] = loaded or fresh(
                    program_id,
                    goal,
                )

                st.session_state[
                    "adaptive_revision"
                ] = rev

                st.session_state[
                    "adaptive_active_id"
                ] = sid

                st.session_state[
                    "adaptive_last"
                ] = None

                st.session_state[
                    "adaptive_unsaved"
                ] = False

                st.rerun()

            except Exception as exc:

                st.error(
                    f"加载失败："
                    f"{type(exc).__name__}: {exc}"
                )

        sid = st.session_state[
            "adaptive_active_id"
        ]

        revision = st.session_state[
            "adaptive_revision"
        ]

        st.caption(
            f"当前会话：{sid}；版本：{revision}"
        )

        if sid != proposed_sid:

            st.warning(
                "输入框的会话ID尚未生效，"
                "请先点击加载。"
            )

        st.caption(
            "教材检索：" + source_status
        )

        if st.session_state[
            "adaptive_unsaved"
        ]:

            st.error(
                "发生保存版本冲突。"
                "请先保留未保存的内容，"
                "然后重新加载会话。"
            )

        # ====================================
        # 6. 用户确认的长期笔记
        # ====================================

        notes = []

        if save:

            st.divider()

            st.subheader(
                "用户确认的长期笔记"
            )

            with st.form(
                "confirm_note",
                clear_on_submit=True,
            ):

                cat = st.selectbox(
                    "类别",
                    options=CATEGORIES,
                )

                content = st.text_area(
                    "需要长期记住的真实体验或目标",
                    max_chars=500,
                )

                submit_note = (
                    st.form_submit_button(
                        "确认并保存笔记"
                    )
                )

            if submit_note:

                try:

                    memory.add_note(
                        sid,
                        cat,
                        content,
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(str(exc))

            notes = memory.list_notes(
                sid,
                limit=50,
            )

            for n in reversed(notes):

                a, b = st.columns(
                    [5, 1]
                )

                a.caption(
                    f"{n['category']}："
                    f"{n['content']}"
                )

                if b.button(
                    "删除",
                    key=f"note-{sid}-{n['note_id']}",
                ):

                    memory.delete_note(
                        sid,
                        n["note_id"],
                    )

                    st.rerun()

            st.divider()

            confirm_delete = st.checkbox(
                "我确认删除当前新版导师会话"
                "及其实践数据",
                value=False,
            )

            if (
                confirm_delete
                and st.button(
                    "删除当前新版会话与实践"
                )
            ):

                try:

                    practices.delete_all_adaptive_session(
                        sid
                    )

                    st.session_state[
                        "adaptive_state"
                    ] = fresh(
                        program_id,
                        goal,
                    )

                    st.session_state[
                        "adaptive_revision"
                    ] = 0

                    st.session_state[
                        "adaptive_last"
                    ] = None

                    st.session_state[
                        "adaptive_unsaved"
                    ] = False

                    st.rerun()

                except Exception as exc:

                    st.error(
                        "删除失败：" + str(exc)
                    )

        else:

            st.info(
                "未授权持久化。可以聊天，"
                "但不能提交长期实践记录。"
            )

    # ========================================
    # 7. 当前会话和目标
    # ========================================

    state = st.session_state[
        "adaptive_state"
    ].model_copy(deep=True)

    state.task.program_id = program_id
    state.task.goal = goal

    # ========================================
    # 8. 自我理解地图
    # ========================================

    if program_id == "self_discovery":

        st.subheader(
            "WHAT / HOW / WHY · 用户确认的线索"
        )

        groups = build_growth_view(
            notes
        )

        cols = st.columns(3)

        for col, group in zip(
            cols,
            groups[:3],
        ):

            with col:

                st.markdown(
                    "**" + group["title"] + "**"
                )

                if not group["entries"]:
                    st.caption(
                        "暂无确认的线索"
                    )

                for item in group["entries"]:

                    st.write(
                        "• " + item
                    )

    # ========================================
    # 9. 导师聊天
    # ========================================

    left, right = st.columns(
        [1.5, 1],
        gap="large",
    )

    with left:

        st.subheader(
            "导师对话"
        )

        for entry in state.history:

            with st.chat_message(
                entry.role
            ):

                st.write(
                    entry.content
                )

        user_message = st.chat_input(
            "提出问题，或要求导师回顾实践反馈"
        )

    # ========================================
    # 10. 查看模型实际做了什么
    # ========================================

    with right:

        st.subheader(
            "模型实际决策"
        )

        turn = st.session_state[
            "adaptive_last"
        ]

        if turn:

            st.write(
                "动作：",
                turn.action,
            )

            st.write(
                "关注：",
                (
                    FOCUS_LABELS.get(
                        turn.focus,
                        "未明确",
                    )
                    if turn.focus
                    else "未明确"
                ),
            )

            st.write(
                "模型建议的方法目录ID：",
                turn.method_id or "无",
            )

            st.caption(
                "方法目录ID不等于已调用原著方法。"
                "没有检索证据时不能作原文归因。"
            )

            for i, event in enumerate(
                turn.trace,
                1,
            ):

                st.write(
                    f"{i}. {event}"
                )

            if (
                turn.action == "propose_experiment"
                and not turn.experiment_protocol
            ):

                st.warning(
                    "这轮没有生成可确认的"
                    "结构化A/B方案。"
                    "可使用下方明确入口。"
                )

            if turn.sources:

                with st.expander(
                    "教材检索候选"
                    "（尚未逐字核验）"
                ):

                    for x in turn.sources:

                        st.write(
                            x.get("source"),
                            x.get("location"),
                        )

                        st.caption(
                            str(
                                x.get("quote", "")
                            )[:240]
                        )

        else:

            st.caption(
                "发送消息后显示模型的实际动作。"
            )

    # ========================================
    # 11. 明确生成A/B方案的入口
    # ========================================

    st.divider()

    st.subheader(
        "明确生成 A/B 对照实践"
    )

    st.caption(
        "当LLM没有正确选择比较模式时，"
        "可以由用户明确指定两个方向。"
        "方案内容仍由Gemma 3生成。"
    )

    with st.form(
        "generate_ab"
    ):

        option_a = st.text_input(
            "方向 A",
            value="算法研究",
        )

        option_b = st.text_input(
            "方向 B",
            value="产品开发",
        )

        submit_ab = (
            st.form_submit_button(
                "由Gemma 3生成待确认方案"
            )
        )

    if submit_ab:

        if not save or sid != proposed_sid:

            st.error(
                "请先开启本地保存并加载"
                "正确会话ID。"
            )

        elif st.session_state[
            "adaptive_unsaved"
        ]:

            st.error(
                "存在保存冲突，请先重新加载。"
            )

        elif mentor.experiment_designer is None:

            st.error(
                "对照实践设计器尚未连接。"
            )

        elif any(
            p["status"] in (
                "proposed",
                "active",
            )
            for p in practices.list_plans(sid)
        ):

            st.error(
                "已有待确认或正在执行的实践，"
                "请先完成或取消。"
            )

        else:

            try:

                with st.spinner(
                    "Gemma 3正在设计两种"
                    "可比较的实践活动……"
                ):

                    plan = (
                        mentor.experiment_designer.build(
                            goal=state.task.goal,

                            history=[
                                m.model_dump()
                                for m in state.history
                            ],

                            draft=(
                                "请针对用户指定的"
                                "两个方向，设计共同背景下"
                                "可比较的实践。"
                            ),

                            options=[
                                option_a.strip(),
                                option_b.strip(),
                            ],
                        )
                    )

                output = render_protocol(
                    plan
                )

                new_state = state.model_copy(
                    deep=True
                )

                new_state.history.append(
                    ChatEntry(
                        role="user",

                        content=(
                            "请设计A/B实践，"
                            f"A={option_a.strip()}，"
                            f"B={option_b.strip()}"
                        ),
                    )
                )

                new_state.history.append(
                    ChatEntry(
                        role="assistant",
                        content=output,
                    )
                )

                turn = MentorTurn(
                    action="propose_experiment",
                    message=output,
                    focus="practice",

                    trace=[
                        "用户明确指定对照方向",
                        "Gemma 3生成结构化对照实践，"
                        "等待用户确认",
                    ],

                    experiment_protocol=(
                        plan.model_dump()
                    ),
                )

                save_turn(
                    new_state,
                    turn,
                    save=save,
                    sid=sid,
                    revision=revision,
                    memory=memory,
                    practices=practices,
                )

                st.rerun()

            except MemoryConflictError as exc:

                st.session_state[
                    "adaptive_unsaved"
                ] = True

                st.error(
                    "会话版本冲突："
                    + str(exc)
                )

            except Exception as exc:

                st.error(
                    "方案生成失败："
                    f"{type(exc).__name__}: {exc}"
                )

    # ========================================
    # 12. 管理实践任务
    # ========================================

    st.divider()

    st.subheader(
        "实践任务与实际执行记录"
    )

    if save:

        plans = practices.list_plans(
            sid
        )

        if not plans:

            st.caption(
                "还没有登记A/B方案。"
                "请先生成，再由你确认。"
            )

        for p in plans[:5]:

            pid = p["plan_id"]

            protocol = p["protocol"]

            title = (
                f"#{pid} · "
                f"{protocol['option_a']} / "
                f"{protocol['option_b']} · "
                f"{p['status']}"
            )

            with st.expander(
                title,

                expanded=(
                    p["status"]
                    in ("proposed", "active")
                ),
            ):

                validated = (
                    ComparisonProtocol.model_validate(
                        protocol
                    )
                )

                st.markdown(
                    render_protocol(validated)
                )

                # ----------------------------
                # 13. 用户确认计划
                # ----------------------------

                if p["status"] == "proposed":

                    if st.button(
                        "我确认开始这份实践",
                        key=f"confirm-{pid}",
                    ):

                        try:

                            practices.confirm(
                                sid,
                                pid,
                            )

                            st.rerun()

                        except Exception as exc:

                            st.error(str(exc))

                # ----------------------------
                # 14. 记录实际活动体验
                # ----------------------------

                if p["status"] == "active":

                    logs = practices.list_logs(
                        sid,
                        pid,
                    )

                    count_a = sum(
                        x["arm"] == "A"
                        for x in logs
                    )

                    count_b = sum(
                        x["arm"] == "B"
                        for x in logs
                    )

                    st.write(
                        f"已记录：A {count_a}次，"
                        f"B {count_b}次。"
                    )

                    st.caption(
                        "建议A→B→B→A；"
                        "真实执行顺序以日志为准。"
                    )

                    with st.form(
                        f"feedback-{pid}",
                        clear_on_submit=True,
                    ):

                        arm = st.radio(
                            "本次活动",
                            options=["A", "B"],
                            horizontal=True,
                        )

                        minutes = st.number_input(
                            "实际投入分钟",
                            min_value=1,
                            max_value=240,
                            value=45,
                            step=5,
                        )

                        willingness = st.slider(
                            "无人要求时愿意再次投入",
                            1,
                            5,
                            3,
                        )

                        engagement = st.slider(
                            "活动过程中的投入感",
                            1,
                            5,
                            3,
                        )

                        reflection = st.text_area(
                            "真实经历：哪些步骤让我"
                            "投入或抗拒？",
                            max_chars=1000,
                        )

                        submit_feedback = (
                            st.form_submit_button(
                                "保存这次真实反馈"
                            )
                        )

                    if submit_feedback:

                        try:

                            practices.add_log(
                                sid,
                                pid,
                                arm,
                                int(minutes),
                                int(willingness),
                                int(engagement),
                                reflection,
                            )

                            st.rerun()

                        except Exception as exc:

                            st.error(str(exc))

                    if (
                        count_a >= 2
                        and count_b >= 2
                        and st.button(
                            "确认两组记录完成",
                            key=f"complete-{pid}",
                        )
                    ):

                        practices.complete(
                            sid,
                            pid,
                        )

                        st.rerun()

                # ----------------------------
                # 15. 展示已记录的反馈
                # ----------------------------

                if p["status"] in (
                    "active",
                    "completed",
                ):

                    for entry in practices.list_logs(
                        sid, pid
                    ):

                        st.caption(
                            f"#{entry['log_id']} "
                            f"{entry['arm']} · "
                            f"{entry['minutes']}分钟 · "
                            f"再次投入"
                            f"{entry['willingness']}/5 · "
                            f"过程投入"
                            f"{entry['engagement']}/5："
                            f"{entry['reflection']}"
                        )

                    st.info(
                        "可以在聊天框中要求导师："
                        "根据已记录的A/B实践反馈，"
                        "分析观察到的差异、干扰因素"
                        "和下一步需要验证的问题。"
                    )

                # ----------------------------
                # 16. 允许取消任务
                # ----------------------------

                if p["status"] in (
                    "proposed",
                    "active",
                ):

                    if st.button(
                        "取消这个方案",
                        key=f"cancel-{pid}",
                    ):

                        practices.cancel(
                            sid,
                            pid,
                        )

                        st.rerun()

    else:

        st.caption(
            "只有在授权本地保存并加载会话后，"
            "才能确认方案与记录实践。"
        )

    # ========================================
    # 17. 正常聊天：加入真实实践反馈
    # ========================================

    if (
        user_message
        and not st.session_state[
            "adaptive_unsaved"
        ]
    ):

        try:

            state.confirmed_notes = (
                [
                    {
                        "category": n["category"],
                        "text": n["content"],
                    }
                    for n in notes[-5:]
                ]

                + practices.reflection_context(
                    sid,
                    limit=6,
                )

                if save
                else []
            )

            with st.spinner(
                "Gemma 3正在结合对话、"
                "记忆和实践反馈决策……"
            ):

                new_state, turn = (
                    mentor.reply(
                        state,
                        user_message,
                    )
                )

            save_turn(
                new_state,
                turn,
                save=save,
                sid=sid,
                revision=revision,
                memory=memory,
                practices=practices,
            )

            st.rerun()

        except MemoryConflictError as exc:

            st.session_state[
                "adaptive_unsaved"
            ] = True

            st.error(
                "会话被另一窗口修改，"
                "本轮未保存："
                + str(exc)
            )

            st.code(
                "用户：" + user_message,
                language="text",
            )

        except Exception as exc:

            st.error(
                "本轮执行失败："
                f"{type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()
