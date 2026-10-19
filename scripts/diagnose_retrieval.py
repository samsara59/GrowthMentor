
"""定位本地知识库检索数量过少的问题。"""

from dataclasses import fields, is_dataclass

from scripts.try_knowledge_retrieve import build_backend


# ==========================================
# 1. 读取任意对象的字段
# ==========================================

def object_fields(obj):

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if is_dataclass(obj):
        return {
            field.name: getattr(obj, field.name)
            for field in fields(obj)
        }

    if hasattr(obj, "__dict__"):
        return vars(obj)

    return {}


# ==========================================
# 2. 检查知识条目标识和来源
# ==========================================

def inspect_item(item):

    data = object_fields(item)

    print(
        "KnowledgeItem字段：",
        list(data.keys()),
    )

    # 只打印标识和来源，不打印教材全文
    keys = [
        "id",
        "item_id",
        "knowledge_id",
        "source_id",
        "source",
        "source_file",
        "source_path",
        "file_name",
        "title",
        "page",
        "location",
    ]

    for key in keys:

        value = data.get(key)

        if isinstance(
            value,
            (str, int, float),
        ):

            print(
                f"  {key}: {str(value)[:120]}"
            )

    for key in (
        "id",
        "item_id",
        "knowledge_id",
    ):

        if data.get(key) is not None:
            return str(data[key])

    return None


# ==========================================
# 3. 对两个不同主题执行相同检索
# ==========================================

def main():

    backend = build_backend()

    queries = [
        "分析力",
        "行动力",
    ]

    identities = {}

    for query in queries:

        print("\n" + "=" * 50)
        print("检索主题：", query)
        print("=" * 50)

        hits = backend.search(
            query=query,
            limit=10,
        )

        print(
            "请求返回上限：10"
        )

        print(
            "实际返回数量：",
            len(hits),
        )

        identities[query] = []

        for index, hit in enumerate(
            hits,
            start=1,
        ):

            print(
                f"\n【候选 {index}】"
            )

            print(
                "向量距离：",
                getattr(hit, "distance", None),
            )

            review = getattr(
                hit,
                "result",
                None,
            )

            if review is None:

                print(
                    "没有ReviewResult"
                )

                continue

            print(
                "审核状态：",
                getattr(review, "status", None),
            )

            item = getattr(
                review,
                "item",
                None,
            )

            if item is None:
                print(
                    "没有KnowledgeItem"
                )
                continue

            identity = inspect_item(
                item
            )

            if identity is not None:

                identities[query].append(
                    identity
                )

    # ======================================
    # 4. 判断是否命中相同KnowledgeItem
    # ======================================

    print("\n========== 检索交集 ==========")

    first = set(
        identities["分析力"]
    )

    second = set(
        identities["行动力"]
    )

    if first and second:

        intersection = first & second

        print(
            "相同知识条目ID数量：",
            len(intersection),
        )

        print(
            "共有条目ID：",
            sorted(intersection),
        )

    else:

        print(
            "暂时无法比较："
            "知识条目没有可识别的ID字段。"
        )

    print(
        "\n诊断结束。"
        "本次没有写入或同步索引。"
    )


if __name__ == "__main__":
    main()
