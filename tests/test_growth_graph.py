
"""第74课：测试LangGraph执行、暂停、恢复和线程隔离。"""

import tempfile
import unittest

from pathlib import Path

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)
from langgraph.types import Command

from app.agent.adaptive_mentor import (
    ChatEntry,
    MentorTurn,
)
from app.agent.growth_graph import (
    build_growth_graph,
)


# ============================================
# 1. 假导师：测试不需要Ollama
# ============================================

class FakeMentor:

    def __init__(self):
        self.calls = 0
        self.fail_once = False

    def reply(self, state, text):

        self.calls += 1

        if self.fail_once:

            self.fail_once = False

            raise RuntimeError(
                "模拟模型临时故障"
            )

        new_state = state.model_copy(
            deep=True
        )

        new_state.history.extend([
            ChatEntry(
                role="user",
                content=text,
            ),

            ChatEntry(
                role="assistant",
                content="导师收到：" + text,
            ),
        ])

        turn = MentorTurn(
            message="导师收到：" + text,
            action="respond",
            focus="general",
            trace=["模拟导师完成回复"],
        )

        return new_state, turn


# ============================================
# 2. 测试输入
# ============================================

def request(text="我真正喜欢什么？"):

    return {
        "question": text,
        "goal": "探索长期成长方向",
        "program_id": "self_discovery",
        "confirmed_notes": [],
    }


def config(thread_id):

    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


# ============================================
# 3. 开始测试
# ============================================

class TestGrowthGraph(unittest.TestCase):

    def setUp(self):

        self.tmp = (
            tempfile.TemporaryDirectory()
        )

        self.db_path = (
            Path(self.tmp.name)
            / "checkpoints.sqlite"
        )

        self.mentor = FakeMentor()

    def tearDown(self):

        # SQLite上下文结束后再清理临时文件，
        # 避免Windows出现WinError 32。
        self.tmp.cleanup()

    # ========================================
    # 测试1：暂停时不能提前调用模型
    # ========================================

    def test_interrupt_before_mentor(self):

        with SqliteSaver.from_conn_string(
            str(self.db_path)
        ) as saver:

            graph = build_growth_graph(
                self.mentor,
                saver,
            )

            cfg = config("test_pause")

            graph.invoke(
                request(),
                config=cfg,
            )

            snapshot = graph.get_state(
                cfg
            )

            self.assertEqual(
                self.mentor.calls,
                0,
            )

            self.assertEqual(
                snapshot.values["phase"],
                "awaiting_confirmation",
            )

            self.assertTrue(
                snapshot.next,
            )

    # ========================================
    # 测试2：关闭数据库再打开后可以恢复
    # ========================================

    def test_resume_after_reopening_database(self):

        cfg = config("test_restart")

        # 第一次运行：
        # 准备完成，暂停等待用户。
        with SqliteSaver.from_conn_string(
            str(self.db_path)
        ) as saver:

            graph = build_growth_graph(
                self.mentor,
                saver,
            )

            graph.invoke(
                request("我想探索兴趣"),
                config=cfg,
            )

        # 这里已经关闭SQLite连接，
        # 模拟退出Python程序。

        # 第二次重新打开同一个检查点文件：
        with SqliteSaver.from_conn_string(
            str(self.db_path)
        ) as saver:

            recovered = build_growth_graph(
                self.mentor,
                saver,
            )

            before = recovered.get_state(
                cfg
            )

            self.assertEqual(
                before.values["question"],
                "我想探索兴趣",
            )

            finished = recovered.invoke(
                Command(resume=True),
                config=cfg,
            )

            self.assertEqual(
                finished["phase"],
                "completed",
            )

            self.assertEqual(
                self.mentor.calls,
                1,
            )

            self.assertIn(
                "导师收到",
                finished["result"],
            )

    # ========================================
    # 测试3：拒绝执行时不能调用模型
    # ========================================

    def test_reject_does_not_call_mentor(self):

        with SqliteSaver.from_conn_string(
            str(self.db_path)
        ) as saver:

            graph = build_growth_graph(
                self.mentor,
                saver,
            )

            cfg = config("test_reject")

            graph.invoke(
                request(),
                config=cfg,
            )

            finished = graph.invoke(
                Command(resume=False),
                config=cfg,
            )

            self.assertEqual(
                finished["phase"],
                "cancelled",
            )

            self.assertEqual(
                self.mentor.calls,
                0,
            )

    # ========================================
    # 测试4：不同thread_id状态隔离
    # ========================================

    def test_separate_threads(self):

        with SqliteSaver.from_conn_string(
            str(self.db_path)
        ) as saver:

            graph = build_growth_graph(
                self.mentor,
                saver,
            )

            one = config("thread_one")
            two = config("thread_two")

            graph.invoke(
                request("问题A"),
                config=one,
            )

            graph.invoke(
                request("问题B"),
                config=two,
            )

            self.assertEqual(
                graph.get_state(
                    one
                ).values["question"],
                "问题A",
            )

            self.assertEqual(
                graph.get_state(
                    two
                ).values["question"],
                "问题B",
            )

    # ========================================
    # 测试5：节点事件可以逐步返回
    # ========================================

    def test_stream_updates(self):

        with SqliteSaver.from_conn_string(
            str(self.db_path)
        ) as saver:

            graph = build_growth_graph(
                self.mentor,
                saver,
            )

            chunks = list(
                graph.stream(
                    request(),
                    config=config("test_stream"),
                    stream_mode="updates",
                )
            )

            self.assertTrue(
                any(
                    "prepare" in chunk
                    for chunk in chunks
                )
            )

            # 此时还没有收到用户确认，
            # 不能执行导师。
            self.assertEqual(
                self.mentor.calls,
                0,
            )

    # ========================================
    # 测试6：同一线程再次运行保留历史
    # ========================================

    def test_second_turn_keeps_history(self):

        with SqliteSaver.from_conn_string(
            str(self.db_path)
        ) as saver:

            graph = build_growth_graph(
                self.mentor,
                saver,
            )

            cfg = config("two_turns")

            graph.invoke(
                request("第一轮问题"),
                config=cfg,
            )

            graph.invoke(
                Command(resume=True),
                config=cfg,
            )

            self.assertEqual(
                len(
                    graph.get_state(
                        cfg
                    ).values["history"]
                ),
                2,
            )

            graph.invoke(
                request("第二轮问题"),
                config=cfg,
            )

            result = graph.invoke(
                Command(resume=True),
                config=cfg,
            )

            self.assertEqual(
                len(result["history"]),
                4,
            )

            self.assertEqual(
                self.mentor.calls,
                2,
            )


if __name__ == "__main__":
    unittest.main()
