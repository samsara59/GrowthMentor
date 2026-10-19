
from app.rag.answerability import (
    guarded_answer_question,
)


class RAGKnowledgeTool:
    """Agent调用教材问答功能的统一入口。"""

    def __init__(
        self,
        retriever,
        generator,
        judge,
    ):
        self.retriever = retriever
        self.generator = generator
        self.judge = judge

    def answer(self, question: str):
        return guarded_answer_question(
            question=question,
            retriever=self.retriever,
            generator=self.generator,
            judge=self.judge,
            top_k=3,
        )
