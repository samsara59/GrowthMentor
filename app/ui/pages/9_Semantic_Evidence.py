
from pathlib import Path
import sys

import streamlit as st


# 将项目根目录加入 Python 导入路径
ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.agent.semantic_evidence import (
    Evidence,
    Claim,
    SemanticEvidenceValidator,
)


st.title("第77课：语义证据验证")
st.caption(
    "第一阶段：验证引用来源。"
    "不依赖任何大语言模型。"
)

st.subheader("1. 输入原始证据")

evidence_type = st.selectbox(
    "证据类型",
    ["textbook", "user_fact", "hypothesis"],
)

evidence_text = st.text_area(
    "原始证据",
    value=(
        "自我效能感可能影响个体"
        "面对困难时的坚持程度。"
    ),
    height=120,
)

st.subheader("2. 输入需要验证的论断")

claim_text = st.text_area(
    "Agent 生成的论断",
    value="自我效能感低的人一定放弃任务。",
    height=100,
)

quote = st.text_area(
    "Agent 引用的原文",
    value=evidence_text,
    height=100,
)

st.subheader("3. 开始验证")

if st.button("验证证据", type="primary"):

    evidence = Evidence(
        evidence_id="demo_001",
        source_type=evidence_type,
        text=evidence_text,
    )

    claim = Claim(
        text=claim_text,
        evidence_id="demo_001",
        quote=quote,
        source_type=evidence_type,
        proposed_verdict="uncertain",
    )

    validator = SemanticEvidenceValidator(
        [evidence]
    )

    result = validator.validate(claim)

    st.write("来源是否有效：", result.source_valid)
    st.write("验证结果：", result.verdict)
    st.write("原因：", result.reason)

    if not result.source_valid:
        st.error("引用来源验证失败。")
    else:
        st.info(
            "引用来源验证通过。"
            "语义是否支持结论仍待验证。"
        )
