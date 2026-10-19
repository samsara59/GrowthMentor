
import math

from app.rag.embedding_client import (
    OllamaEmbeddingClient,
)


def cosine_similarity(a, b):
    """计算两个向量的余弦相似度。"""

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("零向量无法计算余弦相似度")

    return dot / (norm_a * norm_b)


def main():
    client = OllamaEmbeddingClient(
        model="bge-m3",
        num_gpu=0,
        timeout=300,
    )

    query = "阅读时如何找到需要思考的问题？"

    knowledge_texts = [
        "Question 是学习者想到的、或者作者试图回答的问题。",
        "Source 是提供答案的信息来源。",
        "运动之后需要安排合理的休息与营养。",
    ]

    vectors = client.embed(
        [query] + knowledge_texts
    )

    query_vector = vectors[0]

    print("向量维度：", len(query_vector))

    results = []

    for text, vector in zip(
        knowledge_texts,
        vectors[1:],
    ):
        similarity = cosine_similarity(
            query_vector,
            vector,
        )
        results.append((text, similarity))

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    print("\n=== 语义检索排序 ===")

    for text, score in results:
        print(f"相似度：{score:.4f} | {text}")


if __name__ == "__main__":
    main()
