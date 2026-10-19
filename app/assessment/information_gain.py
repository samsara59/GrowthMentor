
"""
Lesson 78: Expected Information Gain.

不依赖大语言模型。
使用离散概率模型进行候选问题选择。
"""

from dataclasses import dataclass
from math import log2


def entropy(probs):
    return -sum(
        p * log2(p)
        for p in probs
        if p > 0
    )


@dataclass
class Question:
    question_id: str
    text: str

    # 行对应潜在状态，列对应可能回答
    # 每一行的概率之和必须为1
    likelihood: list[list[float]]


def validate_question(prior, question):
    if not prior or abs(sum(prior) - 1) > 1e-6:
        raise ValueError("先验概率之和必须为1")

    if any(p < 0 for p in prior):
        raise ValueError("先验概率不能为负")

    if len(question.likelihood) != len(prior):
        raise ValueError("状态数量不匹配")

    n_answers = len(question.likelihood[0])

    if n_answers == 0:
        raise ValueError("问题至少有一个回答")

    for row in question.likelihood:
        if len(row) != n_answers:
            raise ValueError("回答数量不一致")
        if any(x < 0 for x in row):
            raise ValueError("似然概率不能为负")
        if abs(sum(row) - 1) > 1e-6:
            raise ValueError("似然概率之和必须为1")


def answer_probability(prior, question, answer_id):
    return sum(
        prior[z] * question.likelihood[z][answer_id]
        for z in range(len(prior))
    )


def update_belief(prior, question, answer_id):
    validate_question(prior, question)

    n_answers = len(question.likelihood[0])
    if not 0 <= answer_id < n_answers:
        raise ValueError("回答编号无效")

    p_answer = answer_probability(
        prior, question, answer_id
    )

    if p_answer <= 0:
        raise ValueError("该回答的预测概率为0")

    return [
        prior[z]
        * question.likelihood[z][answer_id]
        / p_answer
        for z in range(len(prior))
    ]


def expected_information_gain(prior, question):
    validate_question(prior, question)

    h_before = entropy(prior)
    expected_h_after = 0.0

    n_answers = len(question.likelihood[0])

    for answer_id in range(n_answers):
        p_answer = answer_probability(
            prior, question, answer_id
        )

        if p_answer <= 0:
            continue

        posterior = update_belief(
            prior, question, answer_id
        )

        expected_h_after += (
            p_answer * entropy(posterior)
        )

    return h_before - expected_h_after


def select_question(prior, questions, asked_ids=None):
    asked_ids = set(asked_ids or [])

    available = [
        q for q in questions
        if q.question_id not in asked_ids
    ]

    if not available:
        return None, 0.0

    best = max(
        available,
        key=lambda q: expected_information_gain(
            prior, q
        ),
    )

    return (
        best,
        expected_information_gain(prior, best),
    )


if __name__ == "__main__":

    prior = [0.5, 0.5]

    questions = [
        Question(
            question_id="q1",
            text="遇到困难时，你通常先制定计划吗？",
            likelihood=[
                [0.9, 0.1],
                [0.1, 0.9],
            ],
        ),
        Question(
            question_id="q2",
            text="你最近遇到过困难吗？",
            likelihood=[
                [0.6, 0.4],
                [0.4, 0.6],
            ],
        ),
    ]

    question, gain = select_question(
        prior, questions
    )

    print("当前熵：", round(entropy(prior), 3))
    print("选择问题：", question.text)
    print("预期信息增益：", round(gain, 3))

    posterior = update_belief(
        prior, question, answer_id=0
    )

    print("回答后的概率：", posterior)
