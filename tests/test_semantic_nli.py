
import unittest

from app.agent.semantic_nli import judge_claim


class FakeProvider:
    def __init__(self, response):
        self.response = response

    def chat(self, system_prompt, user_prompt):
        return self.response


class TestSemanticNLI(unittest.TestCase):

    def test_supported(self):
        provider = FakeProvider(
            '{"verdict":"supported","reason":"一致"}'
        )

        result = judge_claim(
            provider,
            "用户说自己完成了练习。",
            "用户完成了练习。",
        )

        self.assertEqual(
            result["verdict"], "supported"
        )

    def test_uncertain(self):
        provider = FakeProvider(
            '{"verdict":"uncertain","reason":"证据不足"}'
        )

        result = judge_claim(
            provider,
            "练习可能有效。",
            "练习对所有人一定有效。",
        )

        self.assertEqual(
            result["verdict"], "uncertain"
        )

    def test_invalid_json(self):
        provider = FakeProvider(
            "this is not JSON"
        )

        with self.assertRaises(ValueError):
            judge_claim(
                provider,
                "证据",
                "论断",
            )


if __name__ == "__main__":
    unittest.main()
