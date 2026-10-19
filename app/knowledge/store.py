
import hashlib
import json
import sqlite3

from contextlib import closing
from pathlib import Path

from app.knowledge.schema import KnowledgeItem
from app.knowledge.review_pipeline import ReviewResult


VALID_STATUSES = {
    "supported",
    "unsupported",
    "uncertain",
    "evidence_invalid",
    "review_error",
}


def knowledge_id(item: KnowledgeItem) -> str:
    """根据知识内容生成稳定的唯一标识。"""

    content = json.dumps(
        item.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def initialize_store(db_path: str | Path) -> None:
    """初始化数据库及数据表。"""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_reviews (
                    item_id TEXT PRIMARY KEY,
                    concept TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    review_reason TEXT NOT NULL,
                    knowledge_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_review_status
                ON knowledge_reviews(review_status)
                """
            )


def save_review_results(
    db_path: str | Path,
    results: list[ReviewResult],
) -> None:
    """保存知识审核结果，重复知识更新审核状态。"""

    initialize_store(db_path)

    rows = []

    for result in results:
        if result.status not in VALID_STATUSES:
            raise ValueError(
                f"不支持的审核状态: {result.status}"
            )

        rows.append((
            knowledge_id(result.item),
            result.item.concept,
            result.status,
            result.reason,
            result.item.model_dump_json(),
        ))

    with closing(sqlite3.connect(db_path)) as conn:
        with conn:
            conn.executemany(
                """
                INSERT INTO knowledge_reviews (
                    item_id,
                    concept,
                    review_status,
                    review_reason,
                    knowledge_json
                )
                VALUES (?, ?, ?, ?, ?)

                ON CONFLICT(item_id)
                DO UPDATE SET
                    review_status = excluded.review_status,
                    review_reason = excluded.review_reason,
                    updated_at = CURRENT_TIMESTAMP
                """,
                rows,
            )


def load_review_results(
    db_path: str | Path,
    status: str | None = None,
) -> list[ReviewResult]:
    """读取知识，可以根据审核状态筛选。"""

    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"不支持的审核状态: {status}")

    sql = """
        SELECT knowledge_json, review_status, review_reason
        FROM knowledge_reviews
    """

    parameters = ()

    if status is not None:
        sql += " WHERE review_status = ?"
        parameters = (status,)

    sql += " ORDER BY item_id"

    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            sql,
            parameters,
        ).fetchall()

    return [
        ReviewResult(
            item=KnowledgeItem.model_validate_json(
                knowledge_json
            ),
            status=review_status,
            reason=review_reason,
        )
        for knowledge_json, review_status, review_reason in rows
    ]
