
"""自适应成长导师的SQLite长期记忆。"""

import re
import sqlite3

from contextlib import closing
from pathlib import Path

from app.agent.adaptive_mentor import MentorState


_ID_RE = re.compile(
    r"^[a-zA-Z0-9_-]{1,64}$"
)

CATEGORIES = (
    "interest",
    "strength",
    "value",
    "goal",
    "experiment",
    "other",
)


class MemoryConflictError(RuntimeError):
    """另一窗口更新过此会话，禁止旧版本覆盖。"""


def check_session_id(session_id: str) -> str:

    if (
        not isinstance(session_id, str)
        or not _ID_RE.fullmatch(session_id)
    ):
        raise ValueError(
            "会话ID只允许字母、数字、_、-，长度1~64"
        )

    return session_id


class GrowthMemoryStore:

    def __init__(self, db_path: str | Path):

        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 只创建新表，不覆盖旧表
        with closing(self._connect()) as conn:

            with conn:

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS
                    adaptive_mentor_sessions_v1 (
                        session_id TEXT PRIMARY KEY,
                        state_json TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        updated_at TEXT NOT NULL
                            DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS
                    adaptive_growth_notes_v1 (
                        note_id INTEGER PRIMARY KEY
                            AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL
                            DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_adaptive_notes_session
                    ON adaptive_growth_notes_v1(
                        session_id,
                        note_id
                    )
                """)

    def _connect(self):

        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10,
        )

        conn.row_factory = sqlite3.Row

        return conn

    # ========================================
    # 1. 恢复一次成长导师对话
    # ========================================

    def load_session(
        self,
        session_id: str,
    ) -> tuple[MentorState | None, int]:

        session_id = check_session_id(session_id)

        with closing(self._connect()) as conn:

            row = conn.execute(
                "SELECT state_json, revision "
                "FROM adaptive_mentor_sessions_v1 "
                "WHERE session_id=?",
                (session_id,),
            ).fetchone()

        if row is None:
            return None, 0

        state = MentorState.model_validate_json(
            row["state_json"]
        )

        return state, int(row["revision"])

    # ========================================
    # 2. 保存会话：检查版本防止覆盖
    # ========================================

    def save_session(
        self,
        session_id: str,
        state: MentorState,
        expected_revision: int,
    ) -> int:

        session_id = check_session_id(session_id)

        if (
            not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 0
        ):
            raise ValueError(
                "expected_revision必须是非负整数"
            )

        # 不将运行时的confirmed_notes重复写入对话表
        serialized = state.model_dump_json()

        with closing(self._connect()) as conn:

            with conn:

                conn.execute("BEGIN IMMEDIATE")

                row = conn.execute(
                    "SELECT revision "
                    "FROM adaptive_mentor_sessions_v1 "
                    "WHERE session_id=?",
                    (session_id,),
                ).fetchone()

                actual = (
                    int(row["revision"])
                    if row
                    else 0
                )

                if actual != expected_revision:

                    raise MemoryConflictError(
                        "会话已变化："
                        f"当前版本{expected_revision}，"
                        f"数据库版本{actual}。"
                        "请重新载入。"
                    )

                new_revision = actual + 1

                if row is None:

                    conn.execute(
                        "INSERT INTO "
                        "adaptive_mentor_sessions_v1 "
                        "(session_id,state_json,revision) "
                        "VALUES(?,?,?)",
                        (
                            session_id,
                            serialized,
                            new_revision,
                        ),
                    )

                else:

                    conn.execute(
                        "UPDATE "
                        "adaptive_mentor_sessions_v1 "
                        "SET state_json=?,revision=?,"
                        "updated_at=CURRENT_TIMESTAMP "
                        "WHERE session_id=?",
                        (
                            serialized,
                            new_revision,
                            session_id,
                        ),
                    )

        return new_revision

    # ========================================
    # 3. 保存用户亲自确认的成长笔记
    # ========================================

    def add_note(
        self,
        session_id: str,
        category: str,
        content: str,
    ) -> int:

        session_id = check_session_id(session_id)

        if category not in CATEGORIES:
            raise ValueError(
                "无效的笔记类别"
            )

        if (
            not isinstance(content, str)
            or not 1 <= len(content.strip()) <= 500
        ):
            raise ValueError(
                "笔记长度应为1到500个字符"
            )

        with closing(self._connect()) as conn:

            with conn:

                cursor = conn.execute(
                    "INSERT INTO "
                    "adaptive_growth_notes_v1 "
                    "(session_id,category,content) "
                    "VALUES(?,?,?)",
                    (
                        session_id,
                        category,
                        content.strip(),
                    ),
                )

                return int(cursor.lastrowid)

    # ========================================
    # 4. 读取最近的用户确认笔记
    # ========================================

    def list_notes(
        self,
        session_id: str,
        limit: int = 12,
    ) -> list[dict]:

        session_id = check_session_id(session_id)

        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 50
        ):
            raise ValueError(
                "limit必须为1到50"
            )

        with closing(self._connect()) as conn:

            rows = conn.execute(
                "SELECT note_id,category,content,created_at "
                "FROM adaptive_growth_notes_v1 "
                "WHERE session_id=? "
                "ORDER BY note_id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()

        return [
            dict(row)
            for row in reversed(rows)
        ]

    # ========================================
    # 5. 单独删除一条成长笔记
    # ========================================

    def delete_note(
        self,
        session_id: str,
        note_id: int,
    ) -> bool:

        session_id = check_session_id(session_id)

        if (
            not isinstance(note_id, int)
            or isinstance(note_id, bool)
            or note_id < 1
        ):
            raise ValueError("无效笔记ID")

        with closing(self._connect()) as conn:

            with conn:

                cursor = conn.execute(
                    "DELETE FROM adaptive_growth_notes_v1 "
                    "WHERE session_id=? AND note_id=?",
                    (session_id, note_id),
                )

                return cursor.rowcount > 0

    # ========================================
    # 6. 删除新架构中的整段会话
    # ========================================

    def delete_session(
        self,
        session_id: str,
    ) -> None:

        session_id = check_session_id(session_id)

        with closing(self._connect()) as conn:

            with conn:

                conn.execute("BEGIN IMMEDIATE")

                conn.execute(
                    "DELETE FROM adaptive_growth_notes_v1 "
                    "WHERE session_id=?",
                    (session_id,),
                )

                conn.execute(
                    "DELETE FROM adaptive_mentor_sessions_v1 "
                    "WHERE session_id=?",
                    (session_id,),
                )
