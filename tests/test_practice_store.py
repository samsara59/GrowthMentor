
"""第68课：方案确认、实践反馈和会话隔离测试。"""

import sqlite3
import tempfile
import unittest

from contextlib import closing
from pathlib import Path

from app.memory.growth_store import (
    GrowthMemoryStore,
)

from app.memory.practice_store import (
    PracticeStore,
)


def protocol():

    return {
        "option_a": "算法研究",
        "option_b": "产品开发",

        "question": (
            "更愿意主动投入哪一种工作？"
        ),

        "common_problem": (
            "同一冠脉分割任务"
        ),

        "task_a": (
            "分析同一病例的算法错误，"
            "尝试小幅改进并记录结果"
        ),

        "output_a": (
            "算法误差分析和实验报告"
        ),

        "task_b": (
            "围绕同一病例创建供使用者"
            "操作的本地分割界面"
        ),

        "output_b": (
            "可交互的本地原型"
        ),

        "minutes_per_session": 45,

        "constants": [
            "相同病例和数据",
            "相同时间预算",
        ],

        "observations": [
            "再次投入意愿",
            "过程投入感",
            "受阻时的继续意愿",
        ],

        "limitations": (
            "一周体验受熟悉程度、疲劳"
            "和任务难度影响，"
            "不代表永久职业偏好。"
        ),

        "confirmation_question": (
            "愿意用这个共同问题"
            "做四次短实践吗？"
        ),
    }


class PracticeStoreTests(unittest.TestCase):

    def setUp(self):

        self.tmp = (
            tempfile.TemporaryDirectory()
        )

        self.db = (
            Path(self.tmp.name)
            / "user_memory.db"
        )

        # 模拟旧数据库，确保新模块不破坏旧表
        with closing(
            sqlite3.connect(self.db)
        ) as conn:

            with conn:

                conn.execute(
                    "CREATE TABLE legacy_data "
                    "(id INTEGER PRIMARY KEY,"
                    "text TEXT)"
                )

                conn.execute(
                    "INSERT INTO legacy_data(text) "
                    "VALUES(?)",
                    ("preserve",),
                )

        GrowthMemoryStore(
            self.db
        )

        self.store = PracticeStore(
            self.db
        )

    def tearDown(self):

        # 上面所有SQLite连接均显式关闭，
        # 避免Windows出现WinError 32。
        self.tmp.cleanup()

    # ========================================
    # 测试1：先确认，才能执行
    # ========================================

    def test_confirmation_required_before_log(
        self,
    ):

        pid = self.store.create_proposal(
            "alice",
            protocol(),
        )

        self.assertEqual(
            self.store.get_plan(
                "alice", pid
            )["status"],
            "proposed",
        )

        with self.assertRaises(
            ValueError
        ):

            self.store.add_log(
                "alice",
                pid,
                "A",
                45,
                4,
                4,
                "先测试未确认状态",
            )

        self.store.confirm(
            "alice",
            pid,
        )

        self.assertEqual(
            self.store.get_plan(
                "alice", pid
            )["status"],
            "active",
        )

    # ========================================
    # 测试2：两组记录齐全才能完成
    # ========================================

    def test_four_logs_then_complete(self):

        pid = self.store.create_proposal(
            "alice",
            protocol(),
        )

        self.store.confirm(
            "alice", pid
        )

        for arm in [
            "A",
            "B",
            "B",
        ]:

            self.store.add_log(
                "alice",
                pid,
                arm,
                45,
                4,
                4,
                "用户主动记录的过程",
            )

        with self.assertRaises(
            ValueError
        ):

            self.store.complete(
                "alice", pid
            )

        self.store.add_log(
            "alice",
            pid,
            "A",
            45,
            3,
            4,
            "第二次A实践的体验",
        )

        self.store.complete(
            "alice",
            pid,
        )

        self.assertEqual(
            self.store.get_plan(
                "alice", pid
            )["status"],
            "completed",
        )

        with self.assertRaises(
            ValueError
        ):

            self.store.add_log(
                "alice",
                pid,
                "B",
                40,
                5,
                4,
                "完成后不能再追加",
            )

    # ========================================
    # 测试3：会话隔离
    # ========================================

    def test_no_cross_session_access(self):

        pid = self.store.create_proposal(
            "alice",
            protocol(),
        )

        with self.assertRaises(
            ValueError
        ):

            self.store.confirm(
                "bob", pid
            )

        with self.assertRaises(
            ValueError
        ):

            self.store.list_logs(
                "bob", pid
            )

        self.assertEqual(
            self.store.list_plans("bob"),
            [],
        )

    # ========================================
    # 测试4：防止重复创建进行中的实践
    # ========================================

    def test_prevents_duplicate_active_plan(
        self,
    ):

        pid = self.store.create_proposal(
            "alice",
            protocol(),
        )

        with self.assertRaises(
            ValueError
        ):

            self.store.create_proposal(
                "alice",
                protocol(),
            )

        self.store.cancel(
            "alice",
            pid,
        )

        new_id = self.store.create_proposal(
            "alice",
            protocol(),
        )

        self.assertIsInstance(
            new_id,
            int,
        )

    # ========================================
    # 测试5：实际反馈进入模型上下文
    # ========================================

    def test_reflection_context_is_user_reported(
        self,
    ):

        pid = self.store.create_proposal(
            "alice",
            protocol(),
        )

        self.assertEqual(
            self.store.reflection_context(
                "alice"
            ),
            [],
        )

        self.store.confirm(
            "alice", pid
        )

        self.store.add_log(
            "alice",
            pid,
            "A",
            35,
            2,
            4,
            "我更投入分析异常案例",
        )

        context = (
            self.store.reflection_context(
                "alice"
            )
        )

        self.assertIn(
            "未经独立核验",
            context[0]["text"],
        )

        self.assertIn(
            "我更投入分析异常案例",
            context[1]["text"],
        )

    # ========================================
    # 测试6：删除新数据但保留旧表
    # ========================================

    def test_delete_new_data_preserves_legacy(
        self,
    ):

        pid = self.store.create_proposal(
            "alice",
            protocol(),
        )

        self.store.confirm(
            "alice", pid
        )

        self.store.add_log(
            "alice",
            pid,
            "A",
            30,
            3,
            3,
            "一次反馈",
        )

        self.store.delete_all_adaptive_session(
            "alice"
        )

        self.assertEqual(
            self.store.list_plans(
                "alice"
            ),
            [],
        )

        with closing(
            sqlite3.connect(self.db)
        ) as conn:

            row = conn.execute(
                "SELECT text "
                "FROM legacy_data"
            ).fetchone()

        self.assertEqual(
            row[0],
            "preserve",
        )


if __name__ == "__main__":
    unittest.main()
