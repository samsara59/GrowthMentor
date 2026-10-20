
"""第72课：带原始文本核验的导师方法候选页面。"""

from pathlib import Path

import streamlit as st

from app.agent.adaptive_mentor import (
    OllamaJsonModel,
)

from app.agent.evidence_coach import (
    SourceEvidenceSearch,
    list_indexed_sources,
)

from app.capabilities.source_search import (
    LocalSourceSearch,
)

from app.knowledge.source_provenance import (
    SourceCatalog,
)

from app.methods.catalog import (
    build_method_catalog,
)

from app.methods.grounded_pack import (
    GroundedMethodCoach,
)


ROOT = Path(__file__).resolve().parents[3]

VECTOR_DIR = (
    ROOT / "data" / "chroma"
)

COLLECTION = (
    "growthmentor_source_chunks_v1"
)


st.set_page_config(
    page_title="教材方法指导",
    layout="wide",
)


# ============================================
# 1. 复用现有MethodCatalog和知识检索
# ============================================

@st.cache_resource(show_spinner=False)
def services():

    import chromadb

    catalog = SourceCatalog(
        ROOT / "books"
    )

    client = chromadb.PersistentClient(
        path=str(VECTOR_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION
    )

    source_names, complete = (
        list_indexed_sources(
            collection
        )
    )

    search_backend = LocalSourceSearch(
        collection
    )

    search = SourceEvidenceSearch(
        collection,
        search_backend._embed,
    )

    model = OllamaJsonModel(
        model="gemma3:4b"
    )

    methods = build_method_catalog()

    coach = GroundedMethodCoach(
        search,
        catalog,
        model,
        methods,
    )

    return (
        coach,
        methods,
        source_names,
        complete,
        collection.count(),
    )


# ============================================
# 2. 主页面
# ============================================

def main():

    st.title(
        "教材方法指导 · 第72课"
    )

    st.caption(
        "选择方法候选与真实教材 → "
        "原文字符匹配 → "
        "Gemma 3结合证据提出有限指导"
    )

    try:

        (
            coach,
            methods,
            source_names,
            all_scanned,
            count,
        ) = services()

    except Exception as exc:

        st.error(
            "教材索引未就绪："
            f"{type(exc).__name__}: {exc}"
        )

        st.info(
            "请先运行第71课来源审计，"
            "确认原文Collection存在且非空。"
        )

        return

    # ========================================
    # 3. 展示真实索引规模
    # ========================================

    a, b = st.columns(2)

    a.metric(
        "已索引的教材文本块",
        count,
    )

    b.metric(
        "索引中的来源名称",
        len(source_names),
    )

    if not all_scanned:

        st.warning(
            "来源列表只扫描了部分向量。"
        )

    if not source_names:

        st.error(
            "没有可识别的来源文件，"
            "当前不能进行方法归因。"
        )

        return

    # ========================================
    # 4. 使用现有MethodCatalog
    # ========================================

    method_options = {
        m.method_id: m
        for m in methods.all()
    }

    if not method_options:

        st.error(
            "现有方法目录为空，"
            "请先检查MethodCatalog。"
        )

        return

    method_id = st.selectbox(
        "选择现有方法目录项"
        "（不代表已经核验原著）",

        options=list(method_options),

        format_func=lambda key: (
            method_options[key].name
        ),
    )

    method = method_options[
        method_id
    ]

    st.caption(
        f"使用目的：{method.intended_use}"
    )

    st.caption(
        f"目录状态：{method.status}；"
        f"限制：{method.caution}"
    )

    # ========================================
    # 5. 从117份已索引来源中选择真实文件
    # ========================================

    keyword = st.text_input(
        "按文件名筛选实际已索引的教材",
        value="",
    )

    filtered = [
        s for s in source_names

        if (
            keyword.strip().casefold()
            in s.casefold()
        )
    ]

    if not filtered:

        st.warning(
            "没有找到名称符合条件的"
            "已索引教材，请调整关键词。"
        )

        return

    source = st.selectbox(
        "本轮实际检索的来源文件",
        options=filtered,
    )

    question = st.text_area(
        "你当前的成长问题",

        value=(
            "我能处理复杂任务，"
            "但如何区分自然擅长的方式"
            "和真正愿意投入的事情？"
        ),

        max_chars=1500,
    )

    st.warning(
        "用户选择的文件只构成本轮候选来源；"
        "模型和页面不会因此声称"
        "该文件属于某个特定作者，"
        "也不会自动写入长期记忆。"
    )

    # ========================================
    # 6. 实际检索与原文核验
    # ========================================

    if st.button(
        "核验原文并尝试方法指导",
        type="primary",
    ):

        try:

            with st.spinner(
                "检索并比对原始教材；"
                "匹配后调用Gemma 3……"
            ):

                result = coach.run(
                    method_id,
                    source,
                    question,
                    limit=5,
                )

            st.session_state[
                "lesson72_result"
            ] = {
                "request": (
                    method_id,
                    source,
                    question,
                ),

                "value": result,
            }

        except Exception as exc:

            st.error(
                "方法指导未完成："
                f"{type(exc).__name__}: {exc}"
            )

            return

    # ========================================
    # 7. 显示与当前选择对应的结果
    # ========================================

    record = st.session_state.get(
        "lesson72_result"
    )

    if (
        not record
        or record["request"]
        != (
            method_id,
            source,
            question,
        )
    ):

        st.info(
            "选择真实教材并提交问题后，"
            "这里才展示本轮验证结果。"
        )

        return

    result = record["value"]

    st.subheader(
        "一、原文字符核验"
    )

    st.write(
        f"通过匹配："
        f"{len(result.verified)}条；"
        f"未匹配："
        f"{len(result.rejected)}条"
    )

    # ========================================
    # 8. 展示真实匹配的证据
    # ========================================

    for item in result.verified:

        with st.expander(
            f"[{item.id}] "
            f"{item.source} · "
            f"{item.location}"
        ):

            st.caption(
                "原始文件："
                f"{item.original_path}；"
                f"状态：{item.match_status}"
            )

            st.write(
                item.excerpt
            )

    if result.rejected:

        with st.expander(
            "未通过原文匹配的候选"
        ):

            st.json(
                list(result.rejected)
            )

    if result.draft is None:

        st.error(
            "没有通过原文字符核验的片段，"
            "本轮已停止模型方法指导。"
        )

        return

    # ========================================
    # 9. 显示模型的有限方法指导
    # ========================================

    st.subheader(
        "二、Gemma 3的有限方法指导"
    )

    draft = result.draft

    if (
        draft.status
        == "insufficient_evidence"
    ):

        st.warning(
            "模型判断这些片段不足以支持"
            "本次方法应用。"
        )

    else:

        st.write(
            "**本轮使用的候选证据编号：** "
            + ", ".join(
                draft.evidence_ids
            )
        )

    st.markdown(
        "**片段实际表达的内容"
        "（模型概括，仍需人工核对）**"
    )

    st.write(
        draft.reading_of_source
    )

    st.markdown(
        "**结合当前情况的可选建议**"
    )

    st.write(
        draft.contextual_application
    )

    st.markdown(
        "**接下来只问一个问题**"
    )

    st.write(
        draft.one_question
    )

    st.markdown(
        "**限制**"
    )

    st.write(
        draft.limitations
    )

    st.caption(
        "字符匹配证明片段能在选定文件中找到；"
        "不自动证明本次解读语义正确、"
        "作者归属真实或完整方法有效。"
    )


if __name__ == "__main__":
    main()
