
import unittest

from pydantic import ValidationError

from app.domain.contracts import (
    GrowthTask,
    GrowthOutcome,
    GrowthHypothesis,
    UserEvidence,
    KnowledgeEvidence,
)

from app.programs.catalog import (
    build_program_catalog,
)

from app.methods.catalog import (
    build_method_catalog,
)

from app.capabilities.registry import (
    CapabilityRegistry,
    CapabilitySpec,
    build_capability_registry,
)


class TestGrowthArchitecture(unittest.TestCase):

    # ========================================
    # 1. 两种成长目标使用相同协议
    # ========================================

    def test_two_different_goals_share_the_same_contract(
        self,
    ):

        for program in (
            "self_discovery",
            "reflective_decision",
        ):

            task = GrowthTask(
                task_id="t1",
                program_id=program,
                goal="解决当前问题",
            )

            self.assertEqual(
                task.status,
                "open",
            )

    # ========================================
    # 2. 用户证据不能冒充教材证据
    # ========================================

    def test_user_evidence_is_not_knowledge_evidence(
        self,
    ):

        with self.assertRaises(ValidationError):

            UserEvidence(
                evidence_id="u1",
                source_title="一本书",
                quote="教材原文",
            )

    # ========================================
    # 3. 假设不能引用不存在的证据
    # ========================================

    def test_outcome_rejects_missing_evidence_reference(
        self,
    ):

        with self.assertRaises(ValidationError):

            GrowthOutcome(
                task_id="t1",
                response="暂不能确认",

                hypotheses=[
                    GrowthHypothesis(
                        hypothesis_id="h1",
                        statement="候选假设",
                        user_evidence_ids=[
                            "不存在"
                        ],
                        uncertainty="需要更多证据",
                    )
                ],
            )

    # ========================================
    # 4. 用户证据与教材证据可以分别引用
    # ========================================

    def test_outcome_accepts_separate_evidence(
        self,
    ):

        result = GrowthOutcome(
            task_id="t1",
            response="需要继续探索",

            user_evidence=[
                UserEvidence(
                    evidence_id="u1",
                    quote="我愿意主动学习",
                )
            ],

            knowledge_evidence=[
                KnowledgeEvidence(
                    evidence_id="k1",
                    source_id="test-source-id",
                    source_title="测试教材",
                    quote="测试用占位文本",
                )
            ],

            hypotheses=[
                GrowthHypothesis(
                    hypothesis_id="h1",
                    statement="可能喜欢主动探索",
                    user_evidence_ids=["u1"],
                    knowledge_evidence_ids=["k1"],
                    uncertainty="缺少跨情境证据",
                )
            ],
        )

        self.assertEqual(
            result.hypotheses[0].user_evidence_ids,
            ["u1"],
        )

    # ========================================
    # 5. 不同类型的成长任务都已注册
    # ========================================

    def test_programs_do_not_require_new_core(self):

        programs = build_program_catalog()

        self.assertEqual(
            len(programs.all()),
            3,
        )

        self.assertEqual(
            programs.get(
                "historical_learning"
            ).status,
            "planned",
        )

    # ========================================
    # 6. 所有Program引用的方法必须存在
    # ========================================

    def test_all_program_method_references_exist(
        self,
    ):

        programs = build_program_catalog()
        methods = build_method_catalog()

        for program in programs.all():

            for method_id in (
                program.suggested_method_ids
            ):

                self.assertEqual(
                    methods.get(method_id).method_id,
                    method_id,
                )

    # ========================================
    # 7. 不能把未核验教材标记为已完成
    # ========================================

    def test_methods_are_not_misrepresented_as_indexed(
        self,
    ):

        for method in build_method_catalog().all():

            self.assertEqual(
                method.status,
                "catalog_only",
            )

    # ========================================
    # 8. 未绑定的工具不能执行
    # ========================================

    def test_capability_is_not_executable_without_handler(
        self,
    ):

        capabilities = build_capability_registry()

        with self.assertRaises(RuntimeError):

            capabilities.invoke(
                "knowledge.retrieve",
                query="分析力",
            )

    # ========================================
    # 9. 已绑定的工具可以正确调用
    # ========================================

    def test_capability_can_be_bound_and_called(
        self,
    ):

        registry = CapabilityRegistry()

        registry.register(
            CapabilitySpec(
                capability_id="test.echo",
                description="测试处理器",
            ),

            handler=lambda text: (
                f"收到：{text}"
            ),
        )

        self.assertEqual(
            registry.invoke(
                "test.echo",
                text="你好",
            ),
            "收到：你好",
        )

    # ========================================
    # 10. 重复注册应当报错
    # ========================================

    def test_duplicate_capability_is_rejected(
        self,
    ):

        registry = CapabilityRegistry()

        spec = CapabilitySpec(
            capability_id="test.echo",
            description="测试",
        )

        registry.register(spec)

        with self.assertRaises(ValueError):
            registry.register(spec)


if __name__ == "__main__":
    unittest.main()
