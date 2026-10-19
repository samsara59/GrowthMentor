
"""第76课：SQLite恢复、跨Agent交接、证据验证和有限重试。"""

import tempfile
import unittest

from pathlib import Path

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)
from langgraph.types import Command
from pydantic import ValidationError

from app.agent.multi_agent_handoff import (
    Handoff,
    build_multi_agent_graph,
)

from app.agent.supervisor_graph import (
    SupervisorPlan,
    TaskSpec,
    CoachReply,
)

from app.agent.experiment_protocol import (
    ComparisonProtocol,
)

from app.knowledge.source_provenance import (
    SourceCatalog,
)


# ============================================
# 1. 测试用虚构教材
# ============================================

QUOTE = (
    "在没有外部任务要求的情况下，"
    "主动投入研究某个问题，"
    "可以作为进一步探索兴趣的线索。"
) * 3


# ============================================
# 2. 模拟Supervisor
# ============================================

class Planner:

    def __init__(self, tasks):

        self.tasks = tasks
        self.calls = 0

    def plan(
        self,
        question,
        goal,
        notes,
    ):

        self.calls += 1

        return SupervisorPlan(
            rationale=(
                "根据用户当前目标组织必要的任务"
                "并综合其实际结果。"
            ),

            tasks=self.tasks,
        )


# ============================================
# 3. 模拟教材检索
# ============================================

class Registry:

    def __init__(
        self,
        quote=QUOTE,
        fail_once=False,
    ):

        self.quote = quote
        self.fail_once = fail_once
        self.calls = 0

    def is_bound(self, key):

        return (
            key == "knowledge.retrieve"
        )

    def invoke(
        self,
        key,
        *,
        query,
        limit,
    ):

        self.calls += 1

        if (
            self.fail_once
            and self.calls == 1
        ):

            raise RuntimeError(
                "模拟临时检索错误"
            )

        return [{
            "source": "course.txt",

            "location": "paragraph1",

            "quote": self.quote,
        }]


# ============================================
# 4. 模拟Practice Agent
# ============================================

class Designer:

    def __init__(self):
        self.calls = 0

    def build(
        self,
        goal,
        history,
        draft,
        options,
    ):

        self.calls += 1

        return ComparisonProtocol(
            option_a=options[0],
            option_b=options[1],

            question=(
                "更愿意自发投入哪一种具体工作？"
            ),

            common_problem=(
                "在同一图像分割任务下"
                "比较研究与应用开发"
            ),

            task_a=(
                "分析同一病例中的分割失败原因"
                "并记录改进思路"
            ),

            task_b=(
                "制作同一病例的交互界面"
                "并记录实际使用体验"
            ),

            output_a="实验记录",
            output_b="交互原型",

            minutes_per_session=45,

            constants=[
                "相同数据",
                "相同时间预算",
            ],

            observations=[
                "自主投入意愿",
                "受阻反应",
                "过程投入",
            ],

            limitations=(
                "这属于短期实践，"
                "不足以推出长期天赋"
                "或职业结论。"
            ),

            confirmation_question=(
                "你愿意自主安排"
                "这四次可比较的小实践吗？"
            ),
        )


# ============================================
# 5. 模拟Coaching Agent
# ============================================

class Coaching:

    def __init__(self):

        self.calls = []

    def answer(
        self,
        question,
        goal,
        results,
    ):

        self.calls.append(
            results
        )

        evidence_ids = [
            hit["evidence_id"]

            for result in results

            for hit in result.get(
                "evidence",
                [],
            )
        ]

        return CoachReply(
            answer=(
                "基于本轮已验证的证据"
                "和用户问题提供暂时性"
                "成长指导。"
            ),

            evidence_ids=evidence_ids,

            limitations=(
                "原文字符匹配不等于语义支持，"
                "实践方案尚未执行。"
            ),
        )


# ============================================
# 6. 生成任务协议
# ============================================

def k(task_id="k1"):

    return TaskSpec(
        task_id=task_id,

        worker="knowledge",

        objective=(
            "检索当前成长问题有关的"
            "原始教材信息"
        ),

        query=(
            "自发兴趣与任务完成"
        ),
    )


def p():

    return TaskSpec(
        task_id="p1",

        worker="practice",

        objective=(
            "设计两个具体工作方向的"
            "短期实践对照"
        ),

        comparison_options=[
            "算法研究",
            "产品开发",
        ],
    )


def cfg(sid):

    return {
        "configurable": {
            "thread_id": sid
        }
    }


# ============================================
# 7. 正式测试
# ============================================

class TestMultiAgentHandoff(
    unittest.TestCase
):

    def setUp(self):

        self.temp = (
            tempfile.TemporaryDirectory()
        )

        self.folder = Path(
            self.temp.name
        )

        self.db = (
            self.folder
            / "checkpoints.sqlite"
        )

        (
            self.folder / "course.txt"
        ).write_text(
            QUOTE,
            encoding="utf-8",
        )

        self.catalog = SourceCatalog(
            self.folder
        )

        self.registry = Registry()
        self.designer = Designer()
        self.coaching = Coaching()

    def tearDown(self):

        self.temp.cleanup()

    def graph(
        self,
        saver,
        tasks,
    ):

        return build_multi_agent_graph(
            Planner(tasks),
            self.registry,
            self.designer,
            self.coaching,
            self.catalog,
            saver,
        )

    def start(
        self,
        graph,
        sid,
    ):

        graph.invoke(
            {
                "question": "探索自发兴趣",
                "goal": "了解长期成长方向",
            },

            config=cfg(sid),
        )

    # ========================================
    # 测试1：拒绝非法消息路线
    # ========================================

    def test_message_protocol_rejects_illegal_recipient(
        self,
    ):

        with self.assertRaises(
            ValidationError
        ):

            Handoff(
                message_id="k1:1",
                task_id="k1",

                sender="knowledge",

                recipient="supervisor",

                status="ok",
                attempt=1,
            )

    # ========================================
    # 测试2：已验证证据成功交接Coaching
    # ========================================

    def test_verified_handoff_reaches_coaching(
        self,
    ):

        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.graph(
                saver,
                [k()],
            )

            self.start(
                graph,
                "verified",
            )

            result = graph.invoke(
                Command(resume=True),

                config=cfg(
                    "verified"
                ),
            )

            self.assertEqual(
                result["phase"],
                "completed",
            )

            self.assertEqual(
                [
                    x["sender"]
                    for x in result[
                        "handoffs"
                    ]
                ],

                [
                    "knowledge",
                    "evidence",
                ],
            )

            self.assertEqual(
                result["worker_results"][0]
                ["evidence"][0]
                ["verification"],

                "matched_original_text",
            )

            self.assertEqual(
                result["evidence_ids"],
                ["K1E1"],
            )

            self.assertEqual(
                self.registry.calls,
                1,
            )

    # ========================================
    # 测试3：未匹配的原文不交给Coaching
    # ========================================

    def test_unmatched_original_not_forwarded_to_coaching(
        self,
    ):

        self.registry = Registry(
            quote=(
                "这个文本完全不在原始教材里面，"
                "同时足够长可以触发核对逻辑。"
            ) * 3
        )

        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.graph(
                saver,
                [k()],
            )

            self.start(
                graph,
                "not_verified",
            )

            result = graph.invoke(
                Command(resume=True),

                config=cfg(
                    "not_verified"
                ),
            )

            self.assertEqual(
                result["phase"],
                "completed",
            )

            self.assertEqual(
                result["worker_results"][0]
                ["evidence"][0]
                ["verification"],

                "text_not_located",
            )

            self.assertEqual(
                self.coaching.calls[0][0][
                    "evidence"
                ],
                [],
            )

            self.assertEqual(
                result["evidence_ids"],
                [],
            )

    # ========================================
    # 测试4：失败后只重试当前任务
    # ========================================

    def test_failed_worker_retries_only_that_task_once(
        self,
    ):

        self.registry = Registry(
            fail_once=True
        )

        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.graph(
                saver,
                [k(), p()],
            )

            self.start(
                graph,
                "retry",
            )

            result = graph.invoke(
                Command(resume=True),

                config=cfg(
                    "retry"
                ),
            )

            self.assertEqual(
                result["phase"],
                "completed",
            )

            self.assertEqual(
                self.registry.calls,
                2,
            )

            self.assertEqual(
                self.designer.calls,
                1,
            )

            self.assertEqual(
                [
                    m["sender"]

                    for m in result[
                        "handoffs"
                    ]
                ],

                [
                    "knowledge",
                    "evidence",

                    "knowledge",
                    "evidence",

                    "practice",
                ],
            )

            self.assertEqual(
                len(
                    result[
                        "worker_results"
                    ]
                ),
                2,
            )

            self.assertEqual(
                result["attempts"]["k1"],
                2,
            )

    # ========================================
    # 测试5：同一个Worker可完成两个任务
    # ========================================

    def test_two_knowledge_tasks_have_separate_messages(
        self,
    ):

        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.graph(
                saver,

                [
                    k("k1"),
                    k("k2"),
                ],
            )

            self.start(
                graph,
                "two_knowledge",
            )

            result = graph.invoke(
                Command(resume=True),

                config=cfg(
                    "two_knowledge"
                ),
            )

            self.assertEqual(
                [
                    r["task_id"]

                    for r in result[
                        "worker_results"
                    ]
                ],

                [
                    "k1",
                    "k2",
                ],
            )

            self.assertEqual(
                result["evidence_ids"],

                [
                    "K1E1",
                    "K2E1",
                ],
            )

    # ========================================
    # 测试6：用户拒绝，不调用Worker
    # ========================================

    def test_decline_invokes_no_worker(
        self,
    ):

        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.graph(
                saver,
                [k(), p()],
            )

            self.start(
                graph,
                "decline",
            )

            result = graph.invoke(
                Command(resume=False),

                config=cfg(
                    "decline"
                ),
            )

            self.assertEqual(
                result["phase"],
                "cancelled",
            )

            self.assertEqual(
                self.registry.calls,
                0,
            )

            self.assertEqual(
                self.designer.calls,
                0,
            )

    # ========================================
    # 测试7：重开SQLite后继续执行
    # ========================================

    def test_resume_after_reopen_sqlite(
        self,
    ):

        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.graph(
                saver,
                [k()],
            )

            self.start(
                graph,
                "restart",
            )

        # 第一个数据库连接已关闭，
        # 模拟停止程序。

        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.graph(
                saver,
                [k()],
            )

            self.assertEqual(
                graph.get_state(
                    cfg("restart")
                ).values["phase"],

                "awaiting_approval",
            )

            result = graph.invoke(
                Command(resume=True),

                config=cfg(
                    "restart"
                ),
            )

            self.assertEqual(
                result["phase"],
                "completed",
            )


if __name__ == "__main__":
    unittest.main()
