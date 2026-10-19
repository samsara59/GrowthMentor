
import unittest

from app.ui.live_monitor import (
    next_expected_node,
    stream_turn,
)


# ============================================
# 1. 用FakeGraph模拟LangGraph
# ============================================

class FakeGraph:

    def __init__(self, updates):
        self.updates = updates
        self.calls = []

    def stream(
        self,
        inputs,
        *,
        stream_mode,
    ):

        self.calls.append(
            (inputs, stream_mode)
        )

        for event in self.updates:
            yield event


# ============================================
# 2. 测试事件数据
# ============================================

def sample_events():

    return [
        {
            "memory_load": {
                "path": [
                    "memory_load"
                ],
            }
        },

        {
            "supervisor": {
                "route": "assessment",
                "path": [
                    "memory_load",
                    "supervisor",
                ],
            }
        },

        {
            "assessment_agent": {
                "reply": "模拟分析结果",
                "path": [
                    "memory_load",
                    "supervisor",
                    "assessment_agent",
                ],
            }
        },

        {
            "memory_save": {
                "path": [
                    "memory_load",
                    "supervisor",
                    "assessment_agent",
                    "memory_save",
                ],
            }
        },
    ]


# ============================================
# 3. 开始测试
# ============================================

class TestLiveMonitor(unittest.TestCase):

    # ----------------------------------------
    # 测试1：完整节点顺序
    # ----------------------------------------

    def test_complete_stream(self):

        graph = FakeGraph(
            sample_events()
        )

        progress = list(
            stream_turn(
                graph,
                "/compare",
                "lesson54_demo",
            )
        )

        names = [
            item.node
            for item in progress
        ]

        self.assertEqual(
            names,
            [
                "memory_load",
                "supervisor",
                "assessment_agent",
                "memory_save",
            ],
        )

        self.assertTrue(
            progress[-1].saved
        )

        self.assertEqual(
            progress[-1].reply,
            "模拟分析结果",
        )

    # ----------------------------------------
    # 测试2：确实使用updates模式
    # ----------------------------------------

    def test_stream_mode(self):

        graph = FakeGraph(
            sample_events()
        )

        list(
            stream_turn(
                graph,
                "/compare",
                "demo",
            )
        )

        inputs, mode = graph.calls[0]

        self.assertEqual(
            mode,
            "updates",
        )

        self.assertEqual(
            inputs["message"],
            "/compare",
        )

    # ----------------------------------------
    # 测试3：选择下一个节点
    # ----------------------------------------

    def test_next_node(self):

        self.assertEqual(
            next_expected_node(
                "memory_load",
                None,
            ),
            "supervisor",
        )

        self.assertEqual(
            next_expected_node(
                "supervisor",
                "assessment",
            ),
            "assessment_agent",
        )

        self.assertEqual(
            next_expected_node(
                "assessment_agent",
                "assessment",
            ),
            "memory_save",
        )

    # ----------------------------------------
    # 测试4：缺少Memory Save必须失败
    # ----------------------------------------

    def test_missing_memory_save(self):

        incomplete = sample_events()[:-1]

        with self.assertRaises(
            RuntimeError
        ):

            list(
                stream_turn(
                    FakeGraph(incomplete),
                    "/compare",
                    "demo",
                )
            )

    # ----------------------------------------
    # 测试5：缺少Agent回复必须失败
    # ----------------------------------------

    def test_missing_reply(self):

        events = [
            {
                "memory_load": {
                    "path": ["memory_load"]
                }
            },
            {
                "memory_save": {
                    "path": [
                        "memory_load",
                        "memory_save",
                    ]
                }
            },
        ]

        with self.assertRaises(
            RuntimeError
        ):

            list(
                stream_turn(
                    FakeGraph(events),
                    "/status",
                    "demo",
                )
            )

    # ----------------------------------------
    # 测试6：拒绝非法输入和未知节点
    # ----------------------------------------

    def test_invalid_events(self):

        with self.assertRaises(
            ValueError
        ):

            list(
                stream_turn(
                    FakeGraph([]),
                    "",
                    "demo",
                )
            )

        with self.assertRaises(
            ValueError
        ):

            list(
                stream_turn(
                    FakeGraph([
                        {
                            "unknown_agent": {}
                        }
                    ]),
                    "hello",
                    "demo",
                )
            )


if __name__ == "__main__":
    unittest.main()
