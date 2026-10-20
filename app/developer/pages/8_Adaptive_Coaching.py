import streamlit as st
import pandas as pd

st.title("自我探索：动态状态估计")

history = [
    {"round": 0, "z1": 0.50, "z2": 0.50},
    {"round": 1, "z1": 0.90, "z2": 0.10},
]

df = pd.DataFrame(history).set_index("round")

st.line_chart(df)

st.info(
    "曲线表示当前模型的假设概率，"
    "不代表客观人格评估或成长分数。"
)