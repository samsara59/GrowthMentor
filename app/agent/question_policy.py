
"""
Lesson 80: Optional question-selection policy.

只有显式提供概率模型时才使用信息增益。
"""

from app.assessment.information_gain import (
    select_question,
)


class InformationGainPolicy:

    def __init__(
        self,
        prior,
        questions,
        asked_ids=None,
    ):
        self.prior = list(prior)
        self.questions = list(questions)
        self.asked_ids = set(asked_ids or [])

    def __call__(self, state, decision):
        question, gain = select_question(
            self.prior,
            self.questions,
            self.asked_ids,
        )

        # 无可用候选时，保留 LLM 原始问题
        if question is None:
            return decision

        # 只覆盖当前问题，不改变行动类型
        return decision.model_copy(
            update={"message": question.text}
        )
