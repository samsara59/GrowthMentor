
"""
Lesson 77: Semantic Evidence Validation.

Model-independent evidence validation.
No Ollama or external API is required.
"""

from dataclasses import dataclass
from typing import Literal


SourceType = Literal[
    "textbook", "user_fact", "hypothesis"
]

Verdict = Literal[
    "supported", "contradicted", "uncertain"
]


@dataclass
class Evidence:
    evidence_id: str
    source_type: SourceType
    text: str


@dataclass
class Claim:
    text: str
    evidence_id: str
    quote: str
    source_type: SourceType
    proposed_verdict: Verdict


@dataclass
class ValidationResult:
    claim: str
    verdict: Verdict
    reason: str
    source_valid: bool


class SemanticEvidenceValidator:

    def __init__(self, evidence_list):
        self.evidence_map = {
            evidence.evidence_id: evidence
            for evidence in evidence_list
        }

    def validate(self, claim: Claim):

        evidence = self.evidence_map.get(
            claim.evidence_id
        )

        # 1. 检查证据 ID 是否存在
        if evidence is None:
            return ValidationResult(
                claim=claim.text,
                verdict="uncertain",
                reason="找不到对应的证据 ID。",
                source_valid=False,
            )

        # 2. 检查引用原文是否真实存在
        if not claim.quote.strip():
            return ValidationResult(
                claim=claim.text,
                verdict="uncertain",
                reason="没有提供可验证的引用。",
                source_valid=False,
            )

        if claim.quote not in evidence.text:
            return ValidationResult(
                claim=claim.text,
                verdict="uncertain",
                reason="引用内容不在原始证据中。",
                source_valid=False,
            )

        # 3. 检查证据类型是否正确
        if claim.source_type != evidence.source_type:
            return ValidationResult(
                claim=claim.text,
                verdict="uncertain",
                reason="论断与证据的来源类型不一致。",
                source_valid=False,
            )

        # 4. 假设不能自动升级为已确认事实
        if claim.source_type == "hypothesis":
            return ValidationResult(
                claim=claim.text,
                verdict="uncertain",
                reason="模型假设不能视为已验证事实。",
                source_valid=True,
            )

        # 5. 来源通过，但语义结论尚未独立验证
        return ValidationResult(
            claim=claim.text,
            verdict="uncertain",
            reason=(
                "证据来源检查通过；"
                "仍需独立语义推断确认是否支持论断。"
            ),
            source_valid=True,
        )


if __name__ == "__main__":

    evidence = Evidence(
        evidence_id="book_001",
        source_type="textbook",
        text=(
            "自我效能感可能影响个体"
            "面对困难时的坚持程度。"
        ),
    )

    claim = Claim(
        text="自我效能感低的人一定放弃任务。",
        evidence_id="book_001",
        quote=evidence.text,
        source_type="textbook",
        proposed_verdict="supported",
    )

    validator = SemanticEvidenceValidator([evidence])

    result = validator.validate(claim)

    print("论断：", result.claim)
    print("来源有效：", result.source_valid)
    print("验证结果：", result.verdict)
    print("解释：", result.reason)
