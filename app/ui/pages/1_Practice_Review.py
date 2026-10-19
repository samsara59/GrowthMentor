
"""同一Streamlit应用的实践反馈复盘页面。"""

import hashlib
import json
from pathlib import Path

import streamlit as st

from app.agent.adaptive_mentor import (
    OllamaJsonModel,
)

from app.agent.practice_review import (
    PracticeReviewService,
    descriptive_facts,
)

from app.memory.growth_store import (
    GrowthMemoryStore,
    check_session_id,
)

from app.memory.practice_store import (
    PracticeStore,
)


ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "data" / "user_memory.db"

st.set_page_config(
    page_title="成长实践复盘",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def services():

    return (
        PracticeStore(DB),

        GrowthMemoryStore(DB),

        PracticeReviewService(
            OllamaJsonModel(
                model="gemma3:4b"
            )
        ),
    )


# ============================================
# 1. 检测日志是否发生修改
# ============================================

def signature(
    logs: list[dict],
) -> str:

    payload = json.dumps(
        logs,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# ============================================
# 2. 主页面
# ============================================

def main():

    practices, memory, reviewer = services()

    st.title(
        "实践证据复盘 · 第69课"
    )

    st.caption(
        "核对实践日志 → "
        "LLM提出暂时解释 → "
        "用户决定是否保存。"
    )

    initial_sid = st.session_state.get(
        "adaptive_active_id",
        "my_growth",
    )

    sid_text = st.text_input(
        "当前会话ID",
        value=initial_sid,
    )

    consent = st.checkbox(
        "同意读取此会话的本地实践日志，"
        "并交给本机Ollama生成复盘",
        value=False,
    )

    if not consent:

        st.info(
            "先确认是否使用本地实践日志。"
            "尚未读取实践内容或调用LLM。"
        )

        return

    # ========================================
    # 3. 加载已确认的实践
    # ========================================

    try:

        sid = check_session_id(
            sid_text
        )

        plans = [
            p
            for p in practices.list_plans(
                sid,
                limit=50,
            )
            if p["status"] in (
                "active",
                "completed",
            )
        ]

    except Exception as exc:

        st.error(
            "无法读取实践："
            f"{type(exc).__name__}: {exc}"
        )

        return

    if not plans:

        st.info(
            "此会话尚无已确认的实践。"
            "请返回长期成长导师页面，"
            "先生成A/B方案并确认开始。"
        )

        return

    by_id = {
        p["plan_id"]: p
        for p in plans
    }

    plan_id = st.selectbox(
        "选择要复盘的实践",

        options=list(by_id),

        format_func=lambda pid: (
            f"#{pid} · "
            f"{by_id[pid]['protocol']['option_a']} / "
            f"{by_id[pid]['protocol']['option_b']} · "
            f"{by_id[pid]['status']}"
        ),
    )

    plan = by_id[plan_id]

    logs = practices.list_logs(
        sid,
        plan_id,
    )

    current_sig = signature(logs)

    # ========================================
    # 4. 展示实际记录和描述统计
    # ========================================

    st.subheader(
        "一、当前已记录的事实"
    )

    st.caption(
        "这里仅统计用户自述的记录。"
        "评分平均值不是因果证据，"
        "不能直接判定长期偏好。"
    )

    stats = descriptive_facts(
        logs
    )

    labels = {
        "A": plan["protocol"]["option_a"],
        "B": plan["protocol"]["option_b"],
    }

    if stats:

        st.dataframe(
            [
                {
                    "活动": (
                        f"{arm} · {labels[arm]}"
                    ),

                    "次数": data["sessions"],

                    "累计分钟": (
                        data["total_minutes"]
                    ),

                    "平均再次投入意愿(1-5)": (
                        data["mean_willingness"]
                    ),

                    "平均过程投入感(1-5)": (
                        data["mean_engagement"]
                    ),
                }

                for arm, data in stats.items()
            ],

            hide_index=True,
            use_container_width=True,
        )

    for log in logs:

        with st.expander(
            f"日志 #{log['log_id']} · "
            f"活动{log['arm']} · "
            f"{log['minutes']}分钟"
        ):

            st.write(
                "再次投入意愿：",
                log["willingness"],
                "/5",
            )

            st.write(
                "过程投入感：",
                log["engagement"],
                "/5",
            )

            st.write(
                "原始体验：",
                log["reflection"],
            )

    if not all(
        arm in stats
        for arm in ("A", "B")
    ):

        st.warning(
            "A和B至少各需一条实际日志后，"
            "才能进行初步对照复盘。"
        )

        return

    if plan["status"] != "completed":

        st.warning(
            "实践仍在进行中。"
            "现在只能进行阶段性复盘，"
            "不能得出最终偏好结论。"
        )

    # ========================================
    # 5. 实际调用Gemma 3分析
    # ========================================

    st.subheader(
        "二、基于证据生成暂时性解释"
    )

    if st.button(
        "生成有日志编号的暂时性复盘",
        type="primary",
    ):

        try:

            with st.spinner(
                "Gemma 3正在分析日志、"
                "核对证据编号……"
            ):

                report = reviewer.analyze(
                    plan,
                    logs,
                )

            st.session_state[
                "practice_review_draft"
            ] = {
                "sid": sid,
                "plan_id": plan_id,
                "signature": current_sig,
                "report": report.model_dump(),
            }

            st.rerun()

        except Exception as exc:

            st.error(
                "复盘失败："
                f"{type(exc).__name__}: {exc}"
            )

    draft = st.session_state.get(
        "practice_review_draft"
    )

    # 防止将旧会话或旧日志的解释用到新数据
    if (
        not draft
        or draft["sid"] != sid
        or draft["plan_id"] != plan_id
        or draft["signature"] != current_sig
    ):

        st.caption(
            "当前没有与这批日志相匹配的复盘结果。"
        )

        return

    report = draft["report"]

    # ========================================
    # 6. 展示事实、解释、反例和不确定性
    # ========================================

    st.markdown("**总体说明**")

    st.write(
        report["summary"]
    )

    st.markdown(
        "**可回到日志核对的现象**"
    )

    st.write(
        report["observed_pattern"]
    )

    st.caption(
        "引用日志："
        + ", ".join(
            f"#{i}"
            for i in report["observed_log_ids"]
        )
    )

    st.markdown(
        "**暂时性解释（不是事实结论）**"
    )

    st.write(
        report["tentative_interpretation"]
    )

    st.caption(
        "支持记录："
        + ", ".join(
            f"#{i}"
            for i in report["supporting_log_ids"]
        )
    )

    counter_ids = report[
        "counter_log_ids"
    ]

    st.caption(
        "反向线索："
        + (
            ", ".join(
                f"#{i}"
                for i in counter_ids
            )
            if counter_ids
            else "模型未指出，不代表不存在"
        )
    )

    st.markdown(
        "**其他可能的解释**"
    )

    for item in report[
        "alternative_explanations"
    ]:

        st.write("- " + item)

    st.markdown(
        "**目前仍不确定的事情**"
    )

    for item in report[
        "uncertainties"
    ]:

        st.write("- " + item)

    st.markdown(
        "**下一步建议（由你决定）**"
    )

    st.write(
        report["next_step"]
    )

    st.write(
        report["confirmation_question"]
    )

    st.caption(
        "系统检查的是日志编号是否存在，"
        "以及观察是否涉及A、B双方。"
        "编号有效不等于解释在语义上正确，"
        "仍需用户核对。"
    )

    # ========================================
    # 7. 用户确认后才写入长期记忆
    # ========================================

    st.subheader(
        "三、是否保存修正后的理解？"
    )

    suggestion = (
        f"实践#{plan_id}的待验证认识："
        f"{report['tentative_interpretation']} "
        "依据日志："
        + ", ".join(
            f"#{i}"
            for i in report["supporting_log_ids"]
        )
        + "。"
    )

    with st.form(
        f"confirm_review_{sid}_"
        f"{plan_id}_{current_sig[:10]}"
    ):

        edited = st.text_area(
            "请修改、删减或纠正模型的解释",
            value=suggestion[:500],
            max_chars=500,
        )

        agree = st.checkbox(
            "我已核对日志，并同意将"
            "上述文字作为暂时性笔记保存"
        )

        submitted = st.form_submit_button(
            "确认写入长期成长记忆"
        )

    if submitted:

        if not agree:

            st.error(
                "必须由你明确确认后才会保存。"
            )

        elif not edited.strip():

            st.error(
                "请填写或修改要确认的内容。"
            )

        elif signature(
            practices.list_logs(
                sid,
                plan_id,
            )
        ) != current_sig:

            st.error(
                "实践日志已发生变化，"
                "请重新生成复盘。"
            )

        else:

            try:

                note_id = memory.add_note(
                    sid,
                    "experiment",
                    edited.strip(),
                )

                st.success(
                    "已保存为用户确认的成长笔记"
                    f" #{note_id}。"
                    "下一次对话可以使用这条线索。"
                )

            except Exception as exc:

                st.error(
                    "保存失败："
                    f"{type(exc).__name__}: {exc}"
                )


if __name__ == "__main__":
    main()
