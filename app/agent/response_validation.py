"""
Lesson 80: Grounded response validation.

The caller must provide provenance-verified sources.
The model extracts individual claims and performs NLI.
No model service is required at import time.
"""

import json
import re

from app.agent.semantic_nli import judge_claim


VERIFIED_STATUS = "verified_against_original"


def parse_json_object(text):
    cleaned = text.strip()

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

    value = json.loads(cleaned)

    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")

    return value


class GroundedResponseValidator:
    """
    Callable adapter for AdaptiveMentor.

    This validator does not certify provenance itself.
    It accepts only sources previously verified
    by a trusted original-document checker.
    """

    def __init__(self, provider):
        self.provider = provider

    def __call__(self, answer, sources):
        if not answer or not answer.strip():
            return self._uncertain("Empty answer")

        if not sources:
            return self._uncertain("No source evidence")

        # Reject sources without verified provenance.
        for source in sources:
            if not isinstance(source, dict):
                return self._uncertain(
                    "Invalid source record"
                )

            if source.get("verification") != VERIFIED_STATUS:
                return self._uncertain(
                    "Source provenance not verified"
                )

            quote = source.get("quote")
            original = source.get("original_text")

            if (
                not isinstance(quote, str)
                or not isinstance(original, str)
                or not quote.strip()
                or not original.strip()
            ):
                return self._uncertain(
                    "Missing quote or original text"
                )

            if quote not in original:
                return self._uncertain(
                    "Quote not found in original text"
                )

        payload = {
            "answer": answer,
            "sources": [
                {
                    "index": index,
                    "quote": source["quote"],
                }
                for index, source in enumerate(sources)
            ],
        }

        system = """
Extract every independently verifiable factual
claim from the answer.

For each claim choose the evidence source index.
Copy the supporting quote exactly from that source.

Do not invent evidence.
Do not omit factual claims.
Return only JSON:

{
  "claims": [
    {
      "claim": "one factual claim",
      "source_index": 0,
      "quote": "exact evidence quote"
    }
  ]
}
"""

        raw = self.provider.chat(
            system,
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
        )

        extracted = parse_json_object(raw)
        claims = extracted.get("claims")

        if not isinstance(claims, list) or not claims:
            return self._uncertain(
                "No verifiable claims extracted"
            )

        results = []

        for item in claims:
            if not isinstance(item, dict):
                return self._uncertain(
                    "Invalid claim object"
                )

            claim = item.get("claim")
            index = item.get("source_index")
            quote = item.get("quote")

            if (
                not isinstance(claim, str)
                or not claim.strip()
                or type(index) is not int
                or not 0 <= index < len(sources)
                or not isinstance(quote, str)
                or not quote.strip()
            ):
                return self._uncertain(
                    "Invalid claim evidence mapping"
                )

            source = sources[index]

            if quote not in source["quote"]:
                return self._uncertain(
                    "Claim quote not in source"
                )

            verdict = judge_claim(
                self.provider,
                evidence=quote,
                claim=claim,
            )

            results.append({
                "claim": claim,
                "source_index": index,
                "verdict": verdict["verdict"],
                "reason": verdict["reason"],
            })

            if verdict["verdict"] != "supported":
                return {
                    "verdict": "uncertain",
                    "reason": "At least one claim unsupported",
                    "claims": results,
                }

        return {
            "verdict": "supported",
            "reason": (
                "All extracted claims supported "
                "by supplied verified evidence"
            ),
            "claims": results,
        }

    @staticmethod
    def _uncertain(reason):
        return {
            "verdict": "uncertain",
            "reason": reason,
            "claims": [],
        }
