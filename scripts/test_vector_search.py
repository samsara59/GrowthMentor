
import argparse

from pathlib import Path

from app.rag.embedding_client import (
    OllamaEmbeddingClient,
)

from app.rag.vector_store import (
    ChromaKnowledgeStore,
)


ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "data" / "knowledge.db"
VECTOR_PATH = ROOT / "data" / "chroma"


def main():
    # 1. 定义命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default="阅读时如何整理问题？",
    )

    parser.add_argument(
        "--sync",
        action="store_true",
        help="先将 SQLite 知识同步到 Chroma",
    )

    args = parser.parse_args()

    # 2. 检查 SQLite 是否存在
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"知识数据库不存在: {DB_PATH}"
        )

    # 3. 创建 BGE-M3 Embedding 客户端
    embedder = OllamaEmbeddingClient(
        model="bge-m3",
        num_gpu=0,
        timeout=300,
    )

    # 4. 创建 Chroma 向量知识库
    store = ChromaKnowledgeStore(
        db_path=DB_PATH,
        vector_path=VECTOR_PATH,
        embedder=embedder,
    )

    # 5. 首次建立或更新索引
    if args.sync:
        count = store.sync()

        print("已同步知识数量：", count)

    # 6. 根据用户问题进行语义检索
    hits = store.search(
        query=args.query,
        limit=5,
    )

    # 7. 展示检索结果
    print("\n=== 语义检索结果 ===")
    print("用户问题：", args.query)
    print("检索数量：", len(hits))

    for index, hit in enumerate(hits, start=1):
        item = hit.result.item

        print(f"\n--- 结果 {index} ---")
        print("概念：", item.concept)
        print("陈述：", item.statement)
        print("余弦距离：", round(hit.distance, 4))
        print("审核状态：", hit.result.status)

        for evidence in item.evidence:
            print("来源：", evidence.source)
            print("位置：", evidence.location)


if __name__ == "__main__":
    main()
