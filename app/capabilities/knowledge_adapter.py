
"""现有Chroma语义检索的能力适配器。"""

from dataclasses import dataclass
from typing import Any, Protocol

from app.capabilities.registry import (
    CapabilityRegistry,
)


# ============================================
# 1. 定义最小检索接口
# ============================================

class SemanticSearchBackend(Protocol):

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[Any]:
        ...


# ============================================
# 2. 检索结果容器
# ============================================

@dataclass(frozen=True)
class RetrievalBatch:

    query: str

    # 保留现有检索器返回的原始结果对象
    hits: tuple[Any, ...]

    # 尚未完成原文和来源映射核查
    attribution_status: str = "unverified"

    @property
    def count(self) -> int:
        return len(self.hits)


# ============================================
# 3. 检索适配器
# ============================================

class KnowledgeRetrievalAdapter:

    def __init__(
        self,
        backend: SemanticSearchBackend,
    ):
        self.backend = backend

    def __call__(
        self,
        query: str,
        limit: int = 5,
    ) -> RetrievalBatch:

        # A. 验证查询输入
        if not isinstance(query, str):
            raise TypeError(
                "query必须是字符串"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "检索问题不能为空"
            )

        # B. 控制单次检索规模
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 20
        ):
            raise ValueError(
                "limit必须是1到20之间的整数"
            )

        # C. 调用原有Chroma检索实现
        hits = self.backend.search(
            query=query,
            limit=limit,
        )

        # D. 验证返回类型
        if not isinstance(hits, (list, tuple)):
            raise TypeError(
                "底层search必须返回列表或元组"
            )

        # E. 不修改原始搜索结果
        return RetrievalBatch(
            query=query,
            hits=tuple(hits),
        )


# ============================================
# 4. 将检索能力连接到注册表
# ============================================

def bind_knowledge_retrieval(
    registry: CapabilityRegistry,
    backend: SemanticSearchBackend,
) -> None:

    adapter = KnowledgeRetrievalAdapter(
        backend=backend
    )

    registry.bind(
        "knowledge.retrieve",
        adapter,
    )
