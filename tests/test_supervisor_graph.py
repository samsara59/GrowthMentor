
"""第75课：Supervisor规划、调度、审批与恢复测试。"""

import tempfile
import unittest

from pathlib import Path

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)

from langgraph.types import Command

from app.agent.supervisor_graph import (
    TaskSpec,
    SupervisorPlan,
    CoachReply,
    SupervisorPlanner,
    CoachingSynthesizer,
    build_supervisor_graph,
)


# ============================================
# 1. 模拟LLM
# ============================================

class FakeModel:

    def __init__(
        self,
        outputs,
    ):

        self.outputs = list(outputs)

    def generate_json(
        self,
        system,
        payload,
        schema,
    ):

        return self.outputs.pop(0)


# ============================================
# 2. 模拟Supervisor
# ============================================

class FakePlanner:

    def __init__(
        self,
        plan_value,
    ):

        self.plan_value = plan_value
        self.calls = 0

    def plan(
        self,
        question,
        goal,
        notes,
    ):

        self.calls += 1

        return self.plan_value


# ============================================
# 3. 模拟真实检索能力接口
# ============================================

class FakeRegistry:

    def __init__(
        self,
        bound=True,
    ):

        self.bound = bound
        self.calls = []

    def is_bound(
        self,
        key,
    ):

        return (
            self.bound
            and key == "knowledge.retrieve"
        )

    def invoke(
        self,
        key,
        *,
        query,
        limit,
    ):

        self.calls.append(
            query
        )

        return [{
            "source": "测试教材.txt",

            "location": "record=7",

            "quote": (
                "测试用的来源片段"
                "（尚未核验）"
            ),
        }]


# ============================================
# 4. 模拟实践设计器
# ============================================

class FakeDesigner:

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

        from app.agent.experiment_protocol import (
            ComparisonProtocol,
        )

        return ComparisonProtocol(
            option_a=options[0],
            option_b=options[1],

            question=(
                "哪一种活动更愿意持续投入？"
            ),

            common_problem=(
                "同一冠脉分割案例的工作"
            ),

            task_a=(
                "分析一次算法错误，"
                "记录具体原因并验证改进"
            ),

            output_a=(
                "一份可复核的算法记录"
            ),

            task_b=(
                "制作同一案例的可交互界面"
                "并找人试用"
            ),

            output_b=(
                "可操作的小型原型"
            ),

            minutes_per_session=45,

            constants=[
                "相同问题",
                "相同时间",
            ],

            observations=[
                "主动投入意愿",
                "专注感",
                "受阻后的反应",
            ],

            limitations=(
                "这是短期自述比较，"
                "不能直接推断长期职业倾向。"
            ),

            confirmation_question=(
                "是否愿意自行安排"
                "这两组活动？"
            ),
        )


# ============================================
# 5. 模拟Coaching Worker
# ============================================

class FakeCoaching:

    def __init__(self):

        self.results = []

    def answer(
        self,
        question,
        goal,
        results,
    ):

        self.results.append(
            results
        )

        ids = [
            evidence["evidence_id"]

            for result in results

            for evidence in result.get(
                "evidence",
                [],
            )
        ]

        return CoachReply(
            answer=(
                "已结合本轮任务成果"
                "给出下一步建议。"
            ),

            evidence_ids=ids,

            limitations=(
                "本轮教材候选尚未完成"
                "原文核验。"
            ),
        )


# ============================================
# 6. 构造不同的测试任务
# ============================================

def make_plan(*workers):

    tasks = []

    for worker in workers:

        if worker == "knowledge":

            tasks.append(
                TaskSpec(
                    task_id="k1",

                    worker="knowledge",

                    objective=(
                        "检索与此成长目标"
                        "相关的教材片段"
                    ),

                    query=(
                        "自发兴趣和学习行为的区别"
                    ),
                )
            )

        else:

            tasks.append(
                TaskSpec(
                    task_id="p1",

                    worker="practice",

                    objective=(
                        "设计两种工作方向"
                        "的对照实践"
                    ),

                    comparison_options=[
                        "算法研究",
                        "产品开发",
                    ],
                )
            )

    return SupervisorPlan(
        rationale=(
            "根据用户目标安排必要的步骤，"
            "之后综合回答。"
        ),

        tasks=tasks,
    )


def config(sid):

    return {
        "configurable": {
            "thread_id": sid,
        }
    }


# ============================================
# 7. 测试类
# ============================================

class TestSupervisorGraph(
    unittest.TestCase
):

    def setUp(self):

        self.tmp = (
            tempfile.TemporaryDirectory()
        )

        self.db = (
            Path(self.tmp.name)
            / "checkpoints.sqlite"
        )

        self.registry = FakeRegistry()

        self.designer = FakeDesigner()

        self.coach = FakeCoaching()

    def tearDown(self):

        self.tmp.cleanup()

    def build(
        self,
        saver,
        plan_value,
    ):

        return build_supervisor_graph(
            FakePlanner(plan_value),
            self.registry,
            self.designer,
            self.coach,
            saver,
        )

    # ========================================
    # 测试1：重复Worker不能同时出现
    # ========================================

    def test_protocol_rejects_duplicate_worker(
        self,
    ):

        task = make_plan(
            "knowledge"
        ).tasks[0]

        with self.assertRaises(
            ValueError
        ):

            SupervisorPlan(
                rationale=(
                    "这是一个可用于验证"
                    "的任务规划说明。"
                ),

                tasks=[
                    task,
                    task,
                ],
            )

    # ========================================
    # 测试2：不完整实践任务不能执行
    # ========================================

    def test_protocol_rejects_incomplete_practice(
        self,
    ):

        with self.assertRaises(
            ValueError
        ):

            TaskSpec(
                task_id="p1",

                worker="practice",

                objective=(
                    "给出可执行的对照实践"
                ),

                comparison_options=[
                    "研究"
                ],
            )

    # ========================================
    # 测试3：Supervisor JSON实际校验
    # ========================================

    def test_supervisor_validates_model_json(
        self,
    ):

        output = make_plan(
            "knowledge"
        ).model_dump()

        result = SupervisorPlanner(
            FakeModel([output])
        ).plan(
            "问题",
            "目标",
            [],
        )

        self.assertEqual(
            result.tasks[0].worker,
            "knowledge",
        )

    # ========================================
    # 测试4：拒绝计划不执行Worker
    # ========================================

    def test_pause_then_decline_no_work(
        self,
    ):

        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.build(
                saver,

                make_plan(
                    "knowledge",
                    "practice",
                ),
            )

            graph.invoke(
                {
                    "question": "探索兴趣",
                    "goal": "成长",
                },

                config("decline"),
            )

            self.assertEqual(
                self.registry.calls,
                [],
            )

            self.assertEqual(
                self.designer.calls,
                0,
            )

            result = graph.invoke(
                Command(resume=False),

                config("decline"),
            )

            self.assertEqual(
                result["phase"],
                "cancelled",
            )

            self.assertEqual(
                self.registry.calls,
                [],
            )

    # ========================================
    # 测试5：两个Worker顺序完成并交接
    # ========================================

    def test_two_workers_and_handoff(
        self,
    ):

        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.build(
                saver,

                make_plan(
                    "knowledge",
                    "practice",
                ),
            )

            graph.invoke(
                {
                    "question": (
                        "比较算法研究和产品开发"
                    ),

                    "goal": (
                        "探索长期兴趣"
                    ),
                },

                config("two_tasks"),
            )

            result = graph.invoke(
                Command(resume=True),

                config("two_tasks"),
            )

            self.assertEqual(
                result["phase"],
                "completed",
            )

            self.assertEqual(
                [
                    r["worker"]

                    for r in result[
                        "worker_results"
                    ]
                ],

                [
                    "knowledge",
                    "practice",
                ],
            )

            self.assertEqual(
                result[
                    "worker_results"
                ][1]["status"],
                "proposed",
            )

            self.assertEqual(
                len(
                    self.registry.calls
                ),
                1,
            )

            self.assertEqual(
                self.designer.calls,
                1,
            )

            self.assertEqual(
                result["evidence_ids"],
                ["K1E1"],
            )

            self.assertEqual(
                len(
                    self.coach.results[0]
                ),
                2,
            )

    # ========================================
    # 测试6：关闭并重开SQLite后能恢复
    # ========================================

    def test_reopen_sqlite_restores_plan(
        self,
    ):

        cfg = config(
            "restart"
        )

        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.build(
                saver,

                make_plan(
                    "knowledge"
                ),
            )

            graph.invoke(
                {
                    "question": "查找教材",
                    "goal": "成长目标",
                },

                cfg,
            )

        # 重新打开相同检查点数据库
        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.build(
                saver,

                make_plan(
                    "knowledge"
                ),
            )

            before = graph.get_state(
                cfg
            )

            self.assertEqual(
                before.values["phase"],
                "awaiting_approval",
            )

            result = graph.invoke(
                Command(resume=True),

                cfg,
            )

            self.assertEqual(
                result["phase"],
                "completed",
            )

    # ========================================
    # 测试7：不能引用虚构教材证据
    # ========================================

    def test_unknown_evidence_id_is_rejected(
        self,
    ):

        output = CoachReply(
            answer=(
                "根据知识片段提出"
                "一个具体后续建议。"
            ),

            evidence_ids=[
                "FAKE"
            ],

            limitations=(
                "当前是未核验的"
                "教材候选片段。"
            ),
        ).model_dump()

        with self.assertRaisesRegex(
            ValueError,
            "不存在",
        ):

            CoachingSynthesizer(
                FakeModel([output]),
                attempts=1,
            ).answer(
                "问题",
                "目标",
                [],
            )

    # ========================================
    # 测试8：简单问题不强制调用工具
    # ========================================

    def test_direct_coaching_without_tool(
        self,
    ):

        with SqliteSaver.from_conn_string(
            str(self.db)
        ) as saver:

            graph = self.build(
                saver,
                make_plan(),
            )

            graph.invoke(
                {
                    "question": (
                        "我希望讨论成长目标"
                    ),

                    "goal": "成长",
                },

                config("direct"),
            )

            result = graph.invoke(
                Command(resume=True),

                config("direct"),
            )

            self.assertEqual(
                result["phase"],
                "completed",
            )

            self.assertEqual(
                result["worker_results"],
                [],
            )

            self.assertEqual(
                self.designer.calls,
                0,
            )


if __name__ == "__main__":
    unittest.main()
