
"""
通过可选语言模型判断证据与论断的关系。
"""

import json
import re

from app.core.model_provider import ModelProvider


SYSTEM_PROMPT = """
你是严格的证据语义推断器。

输入包含证据 evidence 与论断 claim。

只能返回以下三种判断：
supported: 证据足以推出论断。
contradicted: 证据与论断发生明确冲突。
uncertain: 证据不足以确认或否定论断。

注意：
- "可能发生"不能证明"一定发生"。
- 没有证据不能判定为 supported。
- 不能编造用户事实。
- 不要使用输入之外的知识补足证据。
- 必须输出JSON对象。

格式：
{
  "verdict": "uncertain",
  "reason": "判断理由"
}
"""


VALID_VERDICTS = {
    "supported",
    "contradicted",
    "uncertain",
}


def judge_claim(
    provider: ModelProvider,
    evidence: str,
    claim: str,
) -> dict:

    prompt = json.dumps(
        {
            "evidence": evidence,
            "claim": claim,
        },
        ensure_ascii=False,
    )

    raw = provider.chat(
        SYSTEM_PROMPT,
        prompt,
    )

    # 兼容返回 ```json ... ``` 的模型
    cleaned = raw.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    result = json.loads(cleaned)

    if not isinstance(result, dict):
        raise ValueError("语义模型必须返回JSON对象")

    verdict = result.get("verdict")

    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"无效语义判断：{verdict}"
        )

    return {
        "verdict": verdict,
        "reason": str(
            result.get("reason", "")
        ),
    }


def verify_with_model(
    provider,
    source_result,
    evidence_text,
    claim_text,
):
    # 必须先经过上一课的来源检查
    if not source_result.source_valid:
        return {
            "verdict": "uncertain",
            "reason": "原始证据来源无效",
        }

    return judge_claim(
        provider,
        evidence_text,
        claim_text,
    )
