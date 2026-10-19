
from dataclasses import dataclass
from pathlib import Path

from app.knowledge.review_pipeline import ReviewResult
from app.knowledge.store import load_review_results


@dataclass(frozen=True)
class SearchHit:
    """一条知识检索结果。"""

    result: ReviewResult
    score: int


def compute_match_score(
    query: str,
    result: ReviewResult,
) -> int:
    """根据查询词与知识内容的匹配程度评分。"""

    query = query.casefold()

    concept = result.item.concept.casefold()
    statement = result.item.statement.casefold()

    # 概念完全相同
    if query == concept:
        return 3

    # 查询词包含在概念名称中
    if query in concept:
        return 2

    # 查询词包含在知识陈述中
    if query in statement:
        return 1

    # 没有匹配
    return 0


def search_knowledge(
    db_path: str | Path,
    query: str,
    limit: int = 5,
) -> list[SearchHit]:
    """从知识库中检索自动审核通过的知识。"""

    path = Path(db_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"知识数据库不存在: {path}"
        )

    query = query.strip()

    if not query:
        raise ValueError("检索关键词不能为空")

    if limit < 1:
        raise ValueError("limit 必须大于等于 1")

    # 1. 只读取自动审核通过的知识
    reviews = load_review_results(
        path,
        status="supported",
    )

    # 2. 计算匹配分数
    hits = []

    for result in reviews:

        score = compute_match_score(
            query,
            result,
        )

        if score > 0:
            hits.append(
                SearchHit(
                    result=result,
                    score=score,
                )
            )

    # 3. 按相关程度排序
    hits.sort(
        key=lambda hit: (
            -hit.score,
            hit.result.item.concept.casefold(),
            hit.result.item.statement.casefold(),
        )
    )

    # 4. 返回前 limit 条
    return hits[:limit]
