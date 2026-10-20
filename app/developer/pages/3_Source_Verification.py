
"""第71课：教材覆盖和原文核验页面，只读。"""

import json
from pathlib import Path

import streamlit as st

from app.knowledge.source_provenance import (
    SourceCatalog,
    verify_excerpt,
)


ROOT = Path(__file__).resolve().parents[3]

REPORT = (
    ROOT / "data"
    / "source_provenance_report.json"
)

st.set_page_config(
    page_title="教材来源核验",
    layout="wide",
)


def main():

    st.title(
        "教材来源与原文核验"
    )

    st.caption(
        "确认文件来源 → 核对真实片段 → "
        "识别解析问题。不会修改原始教材。"
    )

    # ========================================
    # 1. 读取上一步生成的报告
    # ========================================

    if not REPORT.is_file():

        st.error(
            "找不到来源审计报告，"
            "请先在终端运行下面的命令。"
        )

        st.code(
            'cd D:\\1_studying\\GrowthMentor-Agent\n'
            '& "D:\\envs\\growthmentor\\python.exe" '
            '-m scripts.audit_source_provenance '
            '--samples 20',

            language="powershell",
        )

        return

    try:

        data = json.loads(
            REPORT.read_text(
                encoding="utf-8"
            )
        )

        catalog = SourceCatalog(
            ROOT / "books"
        )

    except Exception as exc:

        st.error(
            "读取报告失败："
            f"{type(exc).__name__}: {exc}"
        )

        return

    # ========================================
    # 2. 显示主要覆盖指标
    # ========================================

    a, b, c, d = st.columns(4)

    a.metric(
        "原始教材",
        data.get(
            "original_file_count", 0
        ),
    )

    b.metric(
        "清洗记录",
        data.get(
            "clean_record_count", 0
        ),
    )

    c.metric(
        "清洗来源文件",
        data.get(
            "clean_unique_original_sources",
            0,
        ),
    )

    d.metric(
        "原文向量",
        data.get(
            "source_index_vector_count",
            0,
        ),
    )

    st.write(
        "索引状态：",
        data.get(
            "source_index_status"
        ),
    )

    st.write(
        "原文索引唯一匹配到的教材文件数：",
        data.get(
            "source_index_unique_original_sources",
            0,
        ),
    )

    st.caption(
        "来源文件数量不等于真实内容覆盖率，"
        "同名文件不会静默合并。"
    )

    # ========================================
    # 3. 未映射的教材
    # ========================================

    with st.expander(
        "没有从清洗记录映射到的原始文件"
    ):

        files = data.get(
            "original_files_not_mapped_from_clean",
            [],
        )

        st.write(
            f"共{len(files)}份。"
            "不能直接认定全部解析失败。"
        )

        st.dataframe(
            [
                {"file": filename}
                for filename in files
            ],
            hide_index=True,
        )

    # ========================================
    # 4. OCR和解析器候选
    # ========================================

    with st.expander(
        "可能需要OCR或额外解析器的文件"
    ):

        st.caption(
            "PDF只检查少量页面。"
            "这里显示的是候选，"
            "不是扫描件的最终认定。"
        )

        st.dataframe(
            data.get(
                "ocr_or_extraction_candidates",
                [],
            ),
            hide_index=True,
        )

    # ========================================
    # 5. 异常来源信息
    # ========================================

    with st.expander(
        "来源字段映射异常"
    ):

        st.json(
            data.get(
                "clean_record_resolution",
                {},
            )
        )

        st.dataframe(
            data.get(
                "clean_unresolved_examples",
                [],
            ),
            hide_index=True,
        )

    # ========================================
    # 6. 原文样本抽查
    # ========================================

    st.subheader(
        "抽样核验Chroma中的真实文本"
    )

    samples = data.get(
        "sample_verification",
        [],
    )

    if not samples:

        st.warning(
            "没有可核验样本。"
            "请检查原文索引，"
            "或重新运行 --samples 20。"
        )

        return

    st.dataframe(
        samples,
        hide_index=True,
    )

    selected = st.selectbox(
        "选择一条实际索引向量",

        options=[
            item["vector_id"]
            for item in samples
        ],
    )

    # ========================================
    # 7. 点击后重新读取真实向量并核验
    # ========================================

    if st.button(
        "重新与原始文件逐字比对"
    ):

        try:

            import chromadb

            client = (
                chromadb.PersistentClient(
                    path=str(
                        ROOT / "data" / "chroma"
                    )
                )
            )

            collection = client.get_collection(
                name=(
                    "growthmentor_source_chunks_v1"
                )
            )

            hit = collection.get(
                ids=[selected],

                include=[
                    "metadatas",
                    "documents",
                ],
            )

            if not hit.get("ids"):

                st.error(
                    "该向量已经不存在。"
                    "请重新生成来源报告。"
                )

                return

            meta = (
                hit.get("metadatas")
                or [{}]
            )[0] or {}

            excerpt = (
                hit.get("documents")
                or [""]
            )[0] or ""

            result = verify_excerpt(
                catalog,
                meta.get(
                    "source", ""
                ),
                excerpt,
            )

            st.write(
                "验证结果：",
                result.status,
            )

            st.write(
                "原始文件：",
                result.resolved_path
                or "未定位",
            )

            st.write(
                "文本位置：",
                result.located_at
                or "未确定",
            )

            st.caption(
                result.explanation
            )

            with st.expander(
                "查看Chroma中实际存储的文本"
            ):

                st.write(
                    excerpt
                )

            if (
                result.status
                == "matched_original_text"
            ):

                st.success(
                    "文本字符序列已在原始文件中找到。"
                    "但模型的语义解释仍需核验。"
                )

            else:

                st.warning(
                    "目前无法证明该片段与"
                    "原始文件全文匹配，"
                    "不要将它标记为已验证引用。"
                )

        except Exception as exc:

            st.error(
                "验证失败："
                f"{type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()
