
"""通过CapabilityRegistry测试本地教材检索。"""

import argparse
from dataclasses import fields, is_dataclass

from app.core.config import get_config

from app.capabilities.registry import (
    build_capability_registry,
)

from app.capabilities.knowledge_adapter import (
    bind_knowledge_retrieval,
)

from app.rag.embedding_client import (
    OllamaEmbeddingClient,
)

from app.rag.vector_store import (
    ChromaKnowledgeStore,
)


# ============================================
# 1. 安全显示对象字段结构
# ============================================

def describe_fields(obj) -> dict[str, str]:
    """
    只返回字段名和字段类型。

    不打印教材全文、用户信息或模型提示词。
    """

    if isinstance(obj, dict):
        data = obj

    elif is_dataclass(obj):
        data = {
            item.name: getattr(obj, item.name)
            for item in fields(obj)
        }

    elif hasattr(obj, "model_dump"):
        data = obj.model_dump()

    elif hasattr(obj, "__dict__"):
        data = vars(obj)

    else:
        return {
            "value": type(obj).__name__
        }

    return {
        str(key): type(value).__name__
        for key, value in data.items()
    }


# ============================================
# 2. 构建真实检索后端
# ============================================

def build_backend():

    config = get_config()

    if not config.knowledge_db.is_file():
        raise FileNotFoundError(
            f"找不到知识数据库：{config.knowledge_db}"
        )

    if not config.vector_dir.is_dir():
        raise FileNotFoundError(
            f"找不到Chroma目录：{config.vector_dir}"
        )

    # 本地BGE-M3生成问题Embedding
    embedder = OllamaEmbeddingClient(
        model=config.embedding_model,
        num_gpu=config.num_gpu,
        timeout=config.timeout_seconds,
    )

    # 使用以前已经开发完成的检索后端
    store = ChromaKnowledgeStore(
        db_path=config.knowledge_db,
        vector_path=config.vector_dir,
        embedder=embedder,
    )

    return store


# ============================================
# 3. 测试真实能力调用
# ============================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default="分析力",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    # A. 构建能力注册表
    registry = build_capability_registry()

    print("======== 检索能力检查 ========")

    print(
        "绑定前：",
        registry.is_bound("knowledge.retrieve"),
    )

    # B. 使用真实Chroma后端
    backend = build_backend()

    # C. 绑定能力
    bind_knowledge_retrieval(
        registry=registry,
        backend=backend,
    )

    print(
        "绑定后：",
        registry.is_bound("knowledge.retrieve"),
    )

    # D. 通过统一接口调用
    result = registry.invoke(
        "knowledge.retrieve",
        query=args.query,
        limit=args.limit,
    )

    print("\n【检索执行结果】")

    print("问题：", result.query)

    print(
        "检索记录数量：",
        result.count,
    )

    print(
        "来源映射状态：",
        result.attribution_status,
    )

    # E. 仅检查结构，不打印私有教材正文
    if result.hits:

        first = result.hits[0]

        print("\n【单条检索结果结构】")

        print(
            "外层类型：",
            type(first).__name__,
        )

        print(
            "外层字段：",
            describe_fields(first),
        )

        # 之前的SemanticHit可能存在result字段
        if hasattr(first, "result"):

            print(
                "result内部类型：",
                type(first.result).__name__,
            )

            print(
                "result内部字段：",
                describe_fields(first.result),
            )

    else:

        print(
            "\n没有检索到可用记录。"
            "需要检查知识审核状态、"
            "索引覆盖情况与Embedding配置。"
        )

    print("\n本次测试没有执行索引同步或写入。")


if __name__ == "__main__":
    main()
