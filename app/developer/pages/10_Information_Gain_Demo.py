
from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.assessment.information_gain import (
    Question,
    entropy,
    select_question,
    update_belief,
)


QUESTIONS = [
    Question(
        "q1",
        "面对困难时，你通常会先制定计划吗？",
        [[0.9, 0.1], [0.1, 0.9]],
    ),
    Question(
        "q2",
        "你最近遇到过困难吗？",
        [[0.6, 0.4], [0.4, 0.6]],
    ),
]


st.title("第78课：自适应追问")
st.caption("教学模拟：概率不是人格诊断结果")


if "lesson78_prior" not in st.session_state:
    st.session_state.lesson78_prior = [0.5, 0.5]

if "lesson78_asked" not in st.session_state:
    st.session_state.lesson78_asked = set()


prior = st.session_state.lesson78_prior

st.subheader("当前假设概率")
st.bar_chart({
    "状态A": prior[0],
    "状态B": prior[1],
})

st.metric(
    "当前信息熵",
    round(entropy(prior), 3),
)

question, gain = select_question(
    prior,
    QUESTIONS,
    st.session_state.lesson78_asked,
)

if question is None:
    st.success("本轮演示结束，没有未回答的问题。")
else:
    st.subheader("下一道问题")
    st.write(question.text)

    st.write(
        "预期信息增益：",
        round(gain, 3),
    )

    choice = st.radio(
        "请选择",
        ["是", "否"],
        key=question.question_id,
    )

    if st.button("提交回答"):
        answer_id = 0 if choice == "是" else 1

        updated = update_belief(
            prior,
            question,
            answer_id,
        )

        st.session_state.lesson78_prior = updated
        st.session_state.lesson78_asked.add(
            question.question_id
        )

        st.rerun()


if st.button("重新开始"):
    st.session_state.lesson78_prior = [0.5, 0.5]
    st.session_state.lesson78_asked = set()
    st.rerun()
