
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb

from app.knowledge.review_pipeline import ReviewResult
from app.knowledge.store import (
    knowledge_id,
    load_review_results,
)


# 1. 定义 Embedding 服务接口
class EmbeddingService(Protocol):
    model: str

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...


# 2. 定义语义检索结果
@dataclass(frozen=True)
class SemanticHit:
    result: ReviewResult
    distance: float


# 3. 定义向量知识库
class ChromaKnowledgeStore:

    def __init__(
        self,
        db_path: str | Path,
        vector_path: str | Path,
        embedder: EmbeddingService,
        collection_name: str = "growthmentor_cosine_v1",
    ):
        self.db_path = Path(db_path)
        self.embedder = embedder

        # 打开本地 Chroma 数据库
        self.client = chromadb.PersistentClient(
            path=str(vector_path)
        )

        # 获取或创建知识向量集合
        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=None,
                metadata={
                    "embedding_model": embedder.model,
                },
                configuration={
                    "hnsw": {
                        "space": "cosine",
                    }
                },
            )
        )

        # 防止混用不同 Embedding 模型
        metadata = self.collection.metadata or {}

        if metadata.get("embedding_model") != embedder.model:
            raise ValueError(
                "Embedding 模型与现有索引不匹配，"
                "请使用新的 collection_name"
            )

    def sync(self, batch_size: int = 16) -> int:
        """将 SQLite 的审核通过知识同步到 Chroma。"""

        if batch_size < 1:
            raise ValueError("batch_size 必须大于等于 1")

        if not self.db_path.is_file():
            raise FileNotFoundError(self.db_path)

        # 1. 读取 SQLite 中的可信候选知识
        reviews = load_review_results(
            self.db_path,
            status="supported",
        )

        # 2. 计算应该存在于向量库的知识 ID
        target_ids = {
            knowledge_id(review.item)
            for review in reviews
        }

        # 3. 分批处理知识
        for start in range(0, len(reviews), batch_size):

            batch = reviews[start:start + batch_size]

            ids = [
                knowledge_id(review.item)
                for review in batch
            ]

            documents = [
                (
                    f"{review.item.concept}\n"
                    f"{review.item.statement}"
                )
                for review in batch
            ]

            # 4. 调用 Embedding 模型
            vectors = self.embedder.embed(documents)

            if len(vectors) != len(ids):
                raise RuntimeError(
                    "向量数量与知识数量不一致"
                )

            # 5. 把 ID、文字和向量保存到 Chroma
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=vectors,
            )

        # 6. 找出向量库里已经失效的知识
        existing_ids = set(
            self.collection.get(include=[])["ids"]
        )

        stale_ids = sorted(
            existing_ids - target_ids
        )

        # 7. 删除失效索引
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        return len(reviews)

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SemanticHit]:
        """根据自然语言问题搜索相关知识。"""

        if not query.strip():
            raise ValueError("查询不能为空")

        if limit < 1:
            raise ValueError("limit 必须大于等于 1")

        if not self.db_path.is_file():
            raise FileNotFoundError(self.db_path)

        # 1. 从 SQLite 读取当前审核通过的知识
        reviews = load_review_results(
            self.db_path,
            status="supported",
        )

        valid_items = {
            knowledge_id(review.item): review
            for review in reviews
        }

        if not valid_items:
            return []

        # 2. 检查 Chroma 中是否已经建立索引
        count = self.collection.count()

        if count == 0:
            return []

        # 3. 将用户问题转成向量
        query_vector = self.embedder.embed(
            [query]
        )[0]

        # 4. 向 Chroma 查询最近邻
        # 第一版检索所有候选，便于完整过滤过期记录
        matches = self.collection.query(
            query_embeddings=[query_vector],
            n_results=count,
            include=["distances"],
        )

        # 5. 根据返回的知识 ID 找回 SQLite 记录
        hits = []

        ids = matches["ids"][0]
        distances = matches["distances"][0]

        for item_id, distance in zip(ids, distances):

            review = valid_items.get(item_id)

            # 已撤销或不存在的知识直接排除
            if review is None:
                continue

            hits.append(
                SemanticHit(
                    result=review,
                    distance=float(distance),
                )
            )

            if len(hits) >= limit:
                break

        return hits
