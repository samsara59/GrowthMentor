
import hashlib
import json
import sqlite3

from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from app.assessment.schema import AssessmentDraft

from app.assessment.cross_schema import (
    CrossAssessmentDraft,
)

from app.memory.schema import (
    AgentSession,
    ExperienceRecord,
    VerificationSession,
    VerificationTurn,
)


VALID_STAGES = {
    "idle",
    "awaiting_situation",
    "awaiting_action",
    "awaiting_outcome",
    "awaiting_verification",
}

EDITABLE_FIELDS = {
    "situation",
    "action",
    "outcome",
}


# ============================================
# 1. 计算原始经历指纹
# ============================================

def experience_fingerprint(
    experiences: Sequence[ExperienceRecord],
) -> str:

    payload = json.dumps(
        [asdict(item) for item in experiences],
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# ============================================
# 2. SQLite长期记忆
# ============================================

class MemoryStore:

    def __init__(self, path: str | Path):

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with closing(
            sqlite3.connect(self.path)
        ) as conn:

            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_sessions (
                        session_id TEXT PRIMARY KEY,
                        snapshot_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                            DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    @staticmethod
    def _check_id(
        session_id: str,
    ) -> None:

        if (
            not isinstance(session_id, str)
            or not session_id.strip()
        ):
            raise ValueError(
                "session_id不能为空"
            )

    # ========================================
    # 3. 使过期的派生分析失效
    # ========================================

    @staticmethod
    def _invalidate_derived(
        session: AgentSession,
    ) -> None:

        session.cross_report = None
        session.cross_report_fingerprint = None
        session.verification = None

        if session.stage == "awaiting_verification":
            session.stage = "idle"

    # ========================================
    # 4. 保存完整会话
    # ========================================

    def save_session(
        self,
        session_id: str,
        session: AgentSession,
    ) -> None:

        self._check_id(session_id)

        if session.stage not in VALID_STAGES:
            raise ValueError(
                f"未知会话阶段：{session.stage}"
            )

        if (
            len(session.experiences)
            != len(session.assessments)
        ):
            raise ValueError(
                "经历和单经历分析数量不一致"
            )

        current = experience_fingerprint(
            session.experiences
        )

        if (
            session.cross_report is not None
            and session.cross_report_fingerprint != current
        ):
            raise ValueError(
                "跨经历报告与当前原始经历不一致，"
                "必须重新/compare"
            )

        if (
            session.verification is not None
            and session.verification.source_fingerprint != current
        ):
            raise ValueError(
                "验证记录与原始经历不一致，"
                "必须重新/compare"
            )

        if (
            session.stage == "awaiting_verification"
            and session.verification is None
        ):
            raise ValueError(
                "验证阶段缺少验证状态"
            )

        # 新增字段，但保留原有快照内容
        snapshot = {
            "version": 2,

            "stage": session.stage,

            "draft": dict(session.draft),

            "experiences": [
                asdict(item)
                for item in session.experiences
            ],

            "assessments": [
                item.model_dump(mode="json")
                if item is not None
                else None
                for item in session.assessments
            ],

            "cross_report": (
                session.cross_report.model_dump(
                    mode="json"
                )
                if session.cross_report is not None
                else None
            ),

            "cross_report_fingerprint": (
                session.cross_report_fingerprint
            ),

            "verification": (
                asdict(session.verification)
                if session.verification is not None
                else None
            ),
        }

        data = json.dumps(
            snapshot,
            ensure_ascii=False,
        )

        with closing(
            sqlite3.connect(self.path)
        ) as conn:

            with conn:
                conn.execute(
                    """
                    INSERT INTO agent_sessions (
                        session_id,
                        snapshot_json
                    )
                    VALUES (?, ?)

                    ON CONFLICT(session_id)
                    DO UPDATE SET
                        snapshot_json = excluded.snapshot_json,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (session_id, data),
                )

    # ========================================
    # 5. 恢复完整会话
    # ========================================

    def load_session(
        self,
        session_id: str,
    ) -> AgentSession:

        self._check_id(session_id)

        with closing(
            sqlite3.connect(self.path)
        ) as conn:

            row = conn.execute(
                """
                SELECT snapshot_json
                FROM agent_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return AgentSession()

        try:
            content = json.loads(row[0])

            # 同时兼容第54课的旧版快照
            if content.get("version") not in (1, 2):
                raise ValueError(
                    "不支持的记忆快照版本"
                )

            stage = content["stage"]

            if stage not in VALID_STAGES:
                raise ValueError(
                    "非法的会话阶段"
                )

            experiences = [
                ExperienceRecord(**item)
                for item in content["experiences"]
            ]

            assessments = [
                AssessmentDraft.model_validate(item)
                if item is not None
                else None
                for item in content["assessments"]
            ]

            if len(experiences) != len(assessments):
                raise ValueError(
                    "经历与分析记录数量不一致"
                )

            # 读取跨经历报告
            raw_report = content.get(
                "cross_report"
            )

            report = (
                CrossAssessmentDraft.model_validate(
                    raw_report
                )
                if raw_report is not None
                else None
            )

            # 读取验证流程
            raw_verification = content.get(
                "verification"
            )

            verification = None

            if raw_verification is not None:

                verification = VerificationSession(
                    candidate_number=raw_verification[
                        "candidate_number"
                    ],
                    candidate_name=raw_verification[
                        "candidate_name"
                    ],
                    source_fingerprint=raw_verification[
                        "source_fingerprint"
                    ],
                    pending_question=raw_verification[
                        "pending_question"
                    ],
                    turns=[
                        VerificationTurn(**item)
                        for item in raw_verification.get(
                            "turns", []
                        )
                    ],
                    pending_answer=raw_verification.get(
                        "pending_answer"
                    ),
                    completed=raw_verification.get(
                        "completed",
                        False,
                    ),
                )

            session = AgentSession(
                stage=stage,
                draft=dict(content["draft"]),
                experiences=experiences,
                assessments=assessments,
                cross_report=report,
                cross_report_fingerprint=content.get(
                    "cross_report_fingerprint"
                ),
                verification=verification,
            )

            # 防止使用与当前经历不一致的旧报告
            current = experience_fingerprint(
                session.experiences
            )

            if (
                (
                    report is not None
                    and session.cross_report_fingerprint
                    != current
                )
                or (
                    verification is not None
                    and verification.source_fingerprint
                    != current
                )
            ):
                self._invalidate_derived(session)

            if (
                session.stage == "awaiting_verification"
                and session.verification is None
            ):
                self._invalidate_derived(session)

            return session

        except (
            KeyError,
            TypeError,
            json.JSONDecodeError,
            ValidationError,
        ) as exc:
            raise ValueError(
                "本地记忆快照损坏或不兼容"
            ) from exc

    # ========================================
    # 6. 更正历史经历
    # ========================================

    def update_experience(
        self,
        session_id: str,
        session: AgentSession,
        number: int,
        field_name: str,
        new_value: str,
    ) -> AgentSession:

        self._check_id(session_id)

        if field_name not in EDITABLE_FIELDS:
            raise ValueError(
                "只允许修改situation、action、outcome"
            )

        if (
            not isinstance(new_value, str)
            or not new_value.strip()
        ):
            raise ValueError(
                "新内容不能为空"
            )

        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or not 1 <= number <= len(session.experiences)
        ):
            raise ValueError(
                "经历编号不存在"
            )

        # 复制对象，避免失败污染原始会话
        updated = deepcopy(session)

        updated.experiences[number - 1] = replace(
            updated.experiences[number - 1],
            **{
                field_name: new_value.strip()
            },
        )

        # 单经历分析基于旧文字，必须失效
        updated.assessments[number - 1] = None

        # 跨经历分析和验证也必须失效
        self._invalidate_derived(updated)

        self.save_session(
            session_id,
            updated,
        )

        return updated

    # ========================================
    # 7. 删除指定经历
    # ========================================

    def delete_experience(
        self,
        session_id: str,
        session: AgentSession,
        number: int,
    ) -> AgentSession:

        self._check_id(session_id)

        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or not 1 <= number <= len(session.experiences)
        ):
            raise ValueError(
                "经历编号不存在"
            )

        updated = deepcopy(session)

        updated.experiences.pop(number - 1)
        updated.assessments.pop(number - 1)

        self._invalidate_derived(updated)

        self.save_session(
            session_id,
            updated,
        )

        return updated

    # ========================================
    # 8. 清除当前会话
    # ========================================

    def forget_session(
        self,
        session_id: str,
    ) -> bool:

        self._check_id(session_id)

        with closing(
            sqlite3.connect(self.path)
        ) as conn:

            with conn:
                cursor = conn.execute(
                    """
                    DELETE FROM agent_sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )

                return cursor.rowcount > 0
