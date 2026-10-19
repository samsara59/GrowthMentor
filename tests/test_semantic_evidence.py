
import unittest

from app.agent.semantic_evidence import (
    Evidence,
    Claim,
    SemanticEvidenceValidator,
)


class TestSemanticEvidence(unittest.TestCase):

    def setUp(self):
        self.evidence = Evidence(
            evidence_id="book_001",
            source_type="textbook",
            text="练习可能改善自我调节能力。",
        )

        self.validator = SemanticEvidenceValidator(
            [self.evidence]
        )

    def test_valid_source_not_auto_supported(self):
        claim = Claim(
            text="练习必然改善能力。",
            evidence_id="book_001",
            quote=self.evidence.text,
            source_type="textbook",
            proposed_verdict="supported",
        )

        result = self.validator.validate(claim)

        self.assertTrue(result.source_valid)
        self.assertEqual(
            result.verdict, "uncertain"
        )

    def test_fake_quote(self):
        claim = Claim(
            text="练习必然改善所有人的能力。",
            evidence_id="book_001",
            quote="研究证明练习一定有效。",
            source_type="textbook",
            proposed_verdict="supported",
        )

        result = self.validator.validate(claim)

        self.assertFalse(result.source_valid)
        self.assertEqual(
            result.verdict, "uncertain"
        )

    def test_wrong_evidence_id(self):
        claim = Claim(
            text="练习有益。",
            evidence_id="missing_001",
            quote=self.evidence.text,
            source_type="textbook",
            proposed_verdict="supported",
        )

        result = self.validator.validate(claim)

        self.assertFalse(result.source_valid)

    def test_hypothesis(self):
        hypothesis = Evidence(
            evidence_id="hyp_001",
            source_type="hypothesis",
            text="用户可能习惯回避困难。",
        )

        validator = SemanticEvidenceValidator(
            [hypothesis]
        )

        claim = Claim(
            text="用户习惯回避困难。",
            evidence_id="hyp_001",
            quote=hypothesis.text,
            source_type="hypothesis",
            proposed_verdict="supported",
        )

        result = validator.validate(claim)

        self.assertTrue(result.source_valid)
        self.assertEqual(
            result.verdict, "uncertain"
        )


if __name__ == "__main__":
    unittest.main()
