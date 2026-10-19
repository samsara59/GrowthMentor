
import unittest

from app.capabilities.registry import (
    CapabilityRegistry,
    CapabilitySpec,
    build_capability_registry,
)

from app.capabilities.knowledge_adapter import (
    KnowledgeRetrievalAdapter,
    bind_knowledge_retrieval,
)


# ============================================
# 1. 构建模拟搜索后端
# ============================================

class FakeSearchBackend:

    def __init__(self):
        self.calls = []

    def search(
        self,
        query: str,
        limit: int = 5,
    ):
        self.calls.append(
            (query, limit)
        )

        return [
            {
                "id": "demo-1",
                "text": "测试用的知识片段",
            }
        ]


# ============================================
# 2. 不应被调用的后端
# ============================================

class NoCallBackend:

    def search(self, query, limit=5):
        raise AssertionError(
            "非法输入不能调用底层检索"
        )


# ============================================
# 3. 返回错误格式的后端
# ============================================

class BadBackend:

    def search(self, query, limit=5):
        return "错误格式"


# ============================================
# 4. 开始测试
# ============================================

class TestKnowledgeCapability(unittest.TestCase):

    def test_builtin_capability_is_declared(self):

        registry = build_capability_registry()

        self.assertFalse(
            registry.is_bound("knowledge.retrieve")
        )

    def test_binding_real_search_interface(self):

        registry = build_capability_registry()
        backend = FakeSearchBackend()

        bind_knowledge_retrieval(
            registry,
            backend,
        )

        self.assertTrue(
            registry.is_bound("knowledge.retrieve")
        )

        result = registry.invoke(
            "knowledge.retrieve",
            query="  分析力  ",
            limit=3,
        )

        self.assertEqual(
            backend.calls,
            [("分析力", 3)],
        )

        self.assertEqual(
            result.query,
            "分析力",
        )

        self.assertEqual(
            result.count,
            1,
        )

    def test_query_cannot_be_empty(self):

        adapter = KnowledgeRetrievalAdapter(
            NoCallBackend()
        )

        with self.assertRaises(ValueError):
            adapter(query="   ")

    def test_limit_must_be_valid(self):

        adapter = KnowledgeRetrievalAdapter(
            NoCallBackend()
        )

        for limit in (
            0,
            -1,
            21,
            True,
        ):
            with self.assertRaises(ValueError):
                adapter(
                    query="分析力",
                    limit=limit,
                )

    def test_search_results_are_not_invented(self):

        adapter = KnowledgeRetrievalAdapter(
            FakeSearchBackend()
        )

        result = adapter(
            query="行动力"
        )

        self.assertEqual(
            result.attribution_status,
            "unverified",
        )

        self.assertEqual(
            result.hits[0]["id"],
            "demo-1",
        )

    def test_bad_backend_result_is_rejected(self):

        adapter = KnowledgeRetrievalAdapter(
            BadBackend()
        )

        with self.assertRaises(TypeError):
            adapter(
                query="分析力"
            )

    def test_unknown_capability_cannot_be_bound(self):

        registry = CapabilityRegistry()

        with self.assertRaises(KeyError):
            registry.bind(
                "unknown",
                lambda: None,
            )

    def test_duplicate_binding_is_rejected(self):

        registry = CapabilityRegistry()

        registry.register(
            CapabilitySpec(
                capability_id="test.search",
                description="测试检索",
            )
        )

        registry.bind(
            "test.search",
            lambda: 1,
        )

        with self.assertRaises(ValueError):
            registry.bind(
                "test.search",
                lambda: 2,
            )


if __name__ == "__main__":
    unittest.main()
