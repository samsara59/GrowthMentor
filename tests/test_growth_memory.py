
"""测试长期成长记忆、会话隔离、删除和LLM读取。"""

import json
import sqlite3
import tempfile
import unittest

from contextlib import closing
from pathlib import Path

from app.agent.adaptive_mentor import (
    MentorState,
    MentorPlanner,
    AdaptiveMentor,
)

from app.domain.contracts import GrowthTask

from app.memory.growth_store import (
    GrowthMemoryStore,
    MemoryConflictError,
)


# ============================================
# 1. 创建测试成长任务
# ============================================

def make_state(
    goal="发现喜欢的事情",
):

    return MentorState(
        task=GrowthTask(
            task_id="t1",
            program_id="self_discovery",
            goal=goal,
        )
    )


# ============================================
# 2. 模拟大语言模型
# ============================================

class FakeModel:

    def __init__(self):

        self.payloads = []

    def decide(
        self,
        system,
        payload,
    ):

        self.payloads.append(
            json.loads(payload)
        )

        return {
            "action": "ask_user",
            "message": "你主动做过什么？",
            "query": "",
            "method_id": None,
        }


# ============================================
# 3. 模拟工具注册表
# ============================================

class FakeRegistry:

    def is_bound(
        self,
        name,
    ):

        return False


# ============================================
# 4. 模拟导师方法目录
# ============================================

class FakeMethods:

    def all(self):

        return []


# ============================================
# 5. 开始测试长期记忆
# ============================================

class TestGrowthMemory(unittest.TestCase):

    # ========================================
    # 每个测试运行前：
    # 创建独立的临时数据库
    # ========================================

    def setUp(self):

        self.temp = (
            tempfile.TemporaryDirectory()
        )

        self.path = (
            Path(self.temp.name)
            / "user_memory.db"
        )

        # 关键修复：
        # closing负责在退出with时调用conn.close()
        #
        # with db只负责提交/回滚事务，
        # 不能代替closing。

        with closing(
            sqlite3.connect(self.path)
        ) as db:

            with db:

                db.execute(
                    "CREATE TABLE existing_legacy "
                    "(id INTEGER PRIMARY KEY, "
                    "val TEXT)"
                )

                db.execute(
                    "INSERT INTO existing_legacy(val) "
                    "VALUES(?)",
                    ("untouched",),
                )

        # 此时上述SQLite连接已经关闭，
        # 再初始化新成长记忆表。

        self.store = GrowthMemoryStore(
            self.path
        )

    # ========================================
    # 每个测试结束后：
    # 删除独立临时目录
    # ========================================

    def tearDown(self):

        # 所有连接必须已经关闭，
        # Windows才能删除数据库文件。

        self.temp.cleanup()

    # ========================================
    # 测试1：保存并恢复会话
    # ========================================

    def test_roundtrip_and_version(self):

        state = make_state()

        self.assertEqual(
            self.store.load_session(
                "first"
            ),
            (None, 0),
        )

        revision = self.store.save_session(
            "first",
            state,
            expected_revision=0,
        )

        self.assertEqual(
            revision,
            1,
        )

        restored, version = (
            self.store.load_session(
                "first"
            )
        )

        self.assertEqual(
            version,
            1,
        )

        self.assertEqual(
            restored.task.goal,
            state.task.goal,
        )

        new_revision = (
            self.store.save_session(
                "first",
                state,
                expected_revision=1,
            )
        )

        self.assertEqual(
            new_revision,
            2,
        )

    # ========================================
    # 测试2：禁止旧版本覆盖新版本
    # ========================================

    def test_prevents_stale_overwrite(self):

        state = make_state()

        self.store.save_session(
            "first",
            state,
            0,
        )

        with self.assertRaises(
            MemoryConflictError
        ):

            self.store.save_session(
                "first",
                state,
                0,
            )

        _, revision = (
            self.store.load_session(
                "first"
            )
        )

        self.assertEqual(
            revision,
            1,
        )

    # ========================================
    # 测试3：不同会话不能串用笔记
    # ========================================

    def test_memory_not_cross_session(self):

        self.store.save_session(
            "first",
            make_state("goal one"),
            0,
        )

        self.store.save_session(
            "second",
            make_state("goal two"),
            0,
        )

        first, _ = self.store.load_session(
            "first"
        )

        second, _ = self.store.load_session(
            "second"
        )

        self.assertEqual(
            first.task.goal,
            "goal one",
        )

        self.assertEqual(
            second.task.goal,
            "goal two",
        )

        self.store.add_note(
            "first",
            "interest",
            "喜欢拆解问题",
        )

        self.assertEqual(
            len(
                self.store.list_notes(
                    "second"
                )
            ),
            0,
        )

    # ========================================
    # 测试4：长期笔记确实进入LLM
    # ========================================

    def test_confirmed_notes_reach_llm(self):

        self.store.add_note(
            "first",
            "interest",
            "愿意主动阅读算法论文",
        )

        notes = self.store.list_notes(
            "first"
        )

        state = make_state()

        state.confirmed_notes = [
            {
                "category": n["category"],
                "text": n["content"],
            }
            for n in notes
        ]

        model = FakeModel()

        agent = AdaptiveMentor(
            MentorPlanner(model),
            FakeRegistry(),
            FakeMethods(),
        )

        updated, turn = agent.reply(
            state,
            "我还是不确定兴趣",
        )

        self.assertEqual(
            turn.action,
            "ask_user",
        )

        # 检查LLM实际收到的上下文
        model_context = model.payloads[0]

        self.assertIn(
            "愿意主动阅读算法论文",
            str(
                model_context[
                    "confirmed_long_term_notes"
                ]
            ),
        )

        # 保存时不重复存储运行时笔记
        self.store.save_session(
            "first",
            updated,
            0,
        )

        loaded, _ = self.store.load_session(
            "first"
        )

        self.assertEqual(
            loaded.confirmed_notes,
            [],
        )

        self.assertEqual(
            len(loaded.history),
            2,
        )

    # ========================================
    # 测试5：删除新记忆不破坏旧数据库表
    # ========================================

    def test_delete_note_and_session(self):

        self.store.save_session(
            "first",
            make_state(),
            0,
        )

        note_id = self.store.add_note(
            "first",
            "value",
            "重视自主学习",
        )

        # 不能删除另一个会话的笔记
        self.assertFalse(
            self.store.delete_note(
                "second",
                note_id,
            )
        )

        self.assertTrue(
            self.store.delete_note(
                "first",
                note_id,
            )
        )

        self.store.add_note(
            "first",
            "goal",
            "尝试一个小项目",
        )

        self.store.delete_session(
            "first"
        )

        self.assertEqual(
            self.store.load_session(
                "first"
            ),
            (None, 0),
        )

        self.assertEqual(
            self.store.list_notes(
                "first"
            ),
            [],
        )

        # 关键修复：
        # 以前使用with sqlite3.connect()
        # 没有确保连接关闭。
        #
        # 现在明确使用closing()。

        with closing(
            sqlite3.connect(self.path)
        ) as db:

            row = db.execute(
                "SELECT val "
                "FROM existing_legacy"
            ).fetchone()

            value = row[0]

        # 此时数据库连接已经关闭
        self.assertEqual(
            value,
            "untouched",
        )

    # ========================================
    # 测试6：非法会话和笔记输入
    # ========================================

    def test_validates_session_and_note(self):

        with self.assertRaises(
            ValueError
        ):

            self.store.load_session(
                "../../escape"
            )

        with self.assertRaises(
            ValueError
        ):

            self.store.add_note(
                "first",
                "diagnosis",
                "内容",
            )

        with self.assertRaises(
            ValueError
        ):

            self.store.add_note(
                "first",
                "other",
                " ",
            )


if __name__ == "__main__":
    unittest.main()
