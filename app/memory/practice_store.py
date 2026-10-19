
"""用户确认的 A/B 成长实践与实际反馈。"""

from contextlib import closing
from pathlib import Path
import json
import sqlite3

from app.agent.experiment_protocol import (
    ComparisonProtocol,
)

from app.memory.growth_store import (
    check_session_id,
)


class PracticeStore:

    def __init__(self, db_path: str | Path):

        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with closing(self._db()) as db:

            with db:

                db.execute("""
                    CREATE TABLE IF NOT EXISTS
                    adaptive_practice_plans_v1 (
                        plan_id INTEGER PRIMARY KEY
                            AUTOINCREMENT,

                        session_id TEXT NOT NULL,

                        protocol_json TEXT NOT NULL,

                        status TEXT NOT NULL
                            CHECK(status IN (
                                'proposed',
                                'active',
                                'completed',
                                'cancelled'
                            )),

                        created_at TEXT NOT NULL
                            DEFAULT CURRENT_TIMESTAMP,

                        updated_at TEXT NOT NULL
                            DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                db.execute("""
                    CREATE TABLE IF NOT EXISTS
                    adaptive_practice_logs_v1 (
                        log_id INTEGER PRIMARY KEY
                            AUTOINCREMENT,

                        plan_id INTEGER NOT NULL,

                        session_id TEXT NOT NULL,

                        arm TEXT NOT NULL
                            CHECK(arm IN ('A','B')),

                        minutes INTEGER NOT NULL
                            CHECK(minutes BETWEEN 1 AND 240),

                        willingness INTEGER NOT NULL
                            CHECK(willingness BETWEEN 1 AND 5),

                        engagement INTEGER NOT NULL
                            CHECK(engagement BETWEEN 1 AND 5),

                        reflection TEXT NOT NULL,

                        created_at TEXT NOT NULL
                            DEFAULT CURRENT_TIMESTAMP,

                        FOREIGN KEY(plan_id)
                            REFERENCES adaptive_practice_plans_v1(plan_id)
                    )
                """)

                db.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_practice_session
                    ON adaptive_practice_plans_v1(
                        session_id,plan_id
                    )
                """)

                db.execute("""
                    CREATE INDEX IF NOT EXISTS
                    idx_practice_log_plan
                    ON adaptive_practice_logs_v1(
                        session_id,plan_id,log_id
                    )
                """)

    # ========================================
    # 1. 数据库连接：必须显式关闭
    # ========================================

    def _db(self):

        db = sqlite3.connect(
            str(self.db_path),
            timeout=10,
        )

        db.row_factory = sqlite3.Row

        db.execute(
            "PRAGMA foreign_keys=ON"
        )

        return db

    # ========================================
    # 2. 新方案先保存为待确认
    # ========================================

    def create_proposal(
        self,
        sid: str,
        protocol: dict,
    ) -> int:

        sid = check_session_id(sid)

        plan = ComparisonProtocol.model_validate(
            protocol
        )

        with closing(self._db()) as db:

            with db:

                db.execute(
                    "BEGIN IMMEDIATE"
                )

                pending = db.execute(
                    "SELECT plan_id FROM "
                    "adaptive_practice_plans_v1 "
                    "WHERE session_id=? "
                    "AND status IN ('proposed','active') "
                    "LIMIT 1",
                    (sid,),
                ).fetchone()

                if pending is not None:

                    raise ValueError(
                        "当前已有待确认或正在执行的实践，"
                        "请先完成或取消。"
                    )

                cursor = db.execute(
                    "INSERT INTO "
                    "adaptive_practice_plans_v1 "
                    "(session_id,protocol_json,status) "
                    "VALUES(?,?,'proposed')",

                    (
                        sid,
                        plan.model_dump_json(),
                    ),
                )

                return int(cursor.lastrowid)

    # ========================================
    # 3. 读取实践任务
    # ========================================

    def list_plans(
        self,
        sid: str,
        limit: int = 10,
    ) -> list[dict]:

        sid = check_session_id(sid)

        if not 1 <= limit <= 50:
            raise ValueError(
                "limit必须在1至50之间"
            )

        with closing(self._db()) as db:

            rows = db.execute(
                "SELECT * FROM "
                "adaptive_practice_plans_v1 "
                "WHERE session_id=? "
                "ORDER BY plan_id DESC LIMIT ?",
                (sid, limit),
            ).fetchall()

        result = []

        for row in rows:

            item = dict(row)

            item["protocol"] = json.loads(
                item.pop("protocol_json")
            )

            result.append(item)

        return result

    def get_plan(
        self,
        sid: str,
        plan_id: int,
    ) -> dict:

        sid = check_session_id(sid)

        with closing(self._db()) as db:

            row = db.execute(
                "SELECT * FROM "
                "adaptive_practice_plans_v1 "
                "WHERE session_id=? AND plan_id=?",
                (sid, plan_id),
            ).fetchone()

        if row is None:

            raise ValueError(
                "没有找到当前会话的实践"
            )

        item = dict(row)

        item["protocol"] = json.loads(
            item.pop("protocol_json")
        )

        return item

    # ========================================
    # 4. 状态转换：使用数据库条件更新
    # ========================================

    def _transition(
        self,
        sid: str,
        plan_id: int,
        old: str,
        new: str,
    ):

        sid = check_session_id(sid)

        with closing(self._db()) as db:

            with db:

                db.execute("BEGIN IMMEDIATE")

                cursor = db.execute(
                    "UPDATE adaptive_practice_plans_v1 "
                    "SET status=?,"
                    "updated_at=CURRENT_TIMESTAMP "
                    "WHERE session_id=? "
                    "AND plan_id=? "
                    "AND status=?",

                    (
                        new,
                        sid,
                        plan_id,
                        old,
                    ),
                )

                if cursor.rowcount != 1:

                    raise ValueError(
                        f"任务状态不是{old}，"
                        "或者任务不存在，不能转换状态。"
                    )

    def confirm(
        self,
        sid: str,
        plan_id: int,
    ):

        self._transition(
            sid,
            plan_id,
            "proposed",
            "active",
        )

    def cancel(
        self,
        sid: str,
        plan_id: int,
    ):

        plan = self.get_plan(
            sid, plan_id
        )

        if plan["status"] not in (
            "proposed",
            "active",
        ):

            raise ValueError(
                "只能取消待确认或正在执行的实践"
            )

        self._transition(
            sid,
            plan_id,
            plan["status"],
            "cancelled",
        )

    # ========================================
    # 5. 只有已确认任务才允许记录反馈
    # ========================================

    def add_log(
        self,
        sid: str,
        plan_id: int,
        arm: str,
        minutes: int,
        willingness: int,
        engagement: int,
        reflection: str,
    ) -> int:

        sid = check_session_id(sid)

        if arm not in ("A", "B"):
            raise ValueError(
                "活动只能是A或B"
            )

        if (
            not isinstance(reflection, str)
            or not 1 <= len(reflection.strip()) <= 1000
        ):

            raise ValueError(
                "反馈长度必须为1至1000字符"
            )

        if any(
            type(x) is not int
            for x in (
                minutes,
                willingness,
                engagement,
            )
        ):

            raise ValueError(
                "时长和评分必须是整数"
            )

        if (
            not 1 <= minutes <= 240
            or not 1 <= willingness <= 5
            or not 1 <= engagement <= 5
        ):

            raise ValueError(
                "时长或体验评分超出范围"
            )

        with closing(self._db()) as db:

            with db:

                db.execute("BEGIN IMMEDIATE")

                row = db.execute(
                    "SELECT status FROM "
                    "adaptive_practice_plans_v1 "
                    "WHERE session_id=? "
                    "AND plan_id=?",
                    (sid, plan_id),
                ).fetchone()

                if (
                    row is None
                    or row["status"] != "active"
                ):

                    raise ValueError(
                        "只能记录已确认、"
                        "正在执行的实践"
                    )

                cursor = db.execute(
                    "INSERT INTO "
                    "adaptive_practice_logs_v1 "
                    "(plan_id,session_id,arm,"
                    "minutes,willingness,engagement,"
                    "reflection) "
                    "VALUES(?,?,?,?,?,?,?)",

                    (
                        plan_id,
                        sid,
                        arm,
                        minutes,
                        willingness,
                        engagement,
                        reflection.strip(),
                    ),
                )

                return int(cursor.lastrowid)

    # ========================================
    # 6. 读取实践日志
    # ========================================

    def list_logs(
        self,
        sid: str,
        plan_id: int,
    ) -> list[dict]:

        self.get_plan(sid, plan_id)

        with closing(self._db()) as db:

            rows = db.execute(
                "SELECT log_id,arm,minutes,"
                "willingness,engagement,"
                "reflection,created_at "
                "FROM adaptive_practice_logs_v1 "
                "WHERE session_id=? "
                "AND plan_id=? "
                "ORDER BY log_id ASC",

                (sid, plan_id),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================
    # 7. 完成前检查A/B记录是否齐全
    # ========================================

    def complete(
        self,
        sid: str,
        plan_id: int,
    ):

        logs = self.list_logs(
            sid, plan_id
        )

        count_a = sum(
            x["arm"] == "A"
            for x in logs
        )

        count_b = sum(
            x["arm"] == "B"
            for x in logs
        )

        if count_a < 2 or count_b < 2:

            raise ValueError(
                "至少记录两次A和两次B，"
                "才能标记已完成。"
            )

        self._transition(
            sid,
            plan_id,
            "active",
            "completed",
        )

    # ========================================
    # 8. 为Gemma 3准备真实反馈上下文
    # ========================================

    def reflection_context(
        self,
        sid: str,
        limit: int = 6,
    ) -> list[dict[str, str]]:

        plans = self.list_plans(
            sid,
            limit=10,
        )

        selected = next(
            (
                p for p in plans
                if p["status"] in (
                    "active",
                    "completed",
                )
            ),
            None,
        )

        if selected is None:
            return []

        logs = self.list_logs(
            sid,
            selected["plan_id"],
        )[-limit:]

        protocol = selected["protocol"]

        result = [{
            "category": "experiment",

            "text": (
                "【实践背景】"
                f"状态={selected['status']}；"
                f"A={protocol['option_a']}；"
                f"B={protocol['option_b']}；"
                "以下为用户自述、未经独立核验；"
                "未完成时不得推断最终偏好。"
            ),
        }]

        for item in logs:

            result.append({
                "category": "experiment",

                "text": (
                    f"【用户自述实践#{item['log_id']}】"
                    f"活动={item['arm']}；"
                    f"用时={item['minutes']}分钟；"
                    f"继续意愿={item['willingness']}/5；"
                    f"投入感={item['engagement']}/5；"
                    f"经历={item['reflection'][:400]}"
                ),
            })

        return result

    # ========================================
    # 9. 删除新版会话中的所有相关数据
    # ========================================

    def delete_all_adaptive_session(
        self,
        sid: str,
    ):

        sid = check_session_id(sid)

        with closing(self._db()) as db:

            with db:

                db.execute("BEGIN IMMEDIATE")

                db.execute(
                    "DELETE FROM adaptive_practice_logs_v1 "
                    "WHERE session_id=?",
                    (sid,),
                )

                db.execute(
                    "DELETE FROM adaptive_practice_plans_v1 "
                    "WHERE session_id=?",
                    (sid,),
                )

                db.execute(
                    "DELETE FROM adaptive_growth_notes_v1 "
                    "WHERE session_id=?",
                    (sid,),
                )

                db.execute(
                    "DELETE FROM adaptive_mentor_sessions_v1 "
                    "WHERE session_id=?",
                    (sid,),
                )
