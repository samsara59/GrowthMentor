
import argparse

from pathlib import Path

from app.rag.embedding_client import (
    OllamaEmbeddingClient,
)
from app.rag.vector_store import (
    ChromaKnowledgeStore,
)
from app.knowledge.ollama_client import OllamaClient
from app.rag.pipeline import (
    RAGDraft,
    answer_question,
)

from app.rag.answerability import (
    CoverageAssessment,
    EvidenceSufficiencyJudge,
    guarded_answer_question,
)

ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "data" / "knowledge.db"
VECTOR_PATH = ROOT / "data" / "chroma"


def main():
    # 1. 接收用户的自然语言问题
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="需要根据教材知识回答的问题",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="可选余弦距离阈值，需通过实验确定",
    )

    args = parser.parse_args()

    if not DB_PATH.is_file():
        raise FileNotFoundError(
            "知识数据库不存在，请先完成 SQLite 入库"
        )

    # 2. 创建 Embedding 客户端
    embedder = OllamaEmbeddingClient(
        model="bge-m3",
        num_gpu=0,
        timeout=300,
    )

    # 3. 打开已有的 Chroma 索引和 SQLite
    retriever = ChromaKnowledgeStore(
        db_path=DB_PATH,
        vector_path=VECTOR_PATH,
        embedder=embedder,
    )

    # 4. 创建用于生成回答的本地大模型
    generator = OllamaClient(
        model="gemma3:4b",
        num_gpu=0,
        timeout=300,
        response_format=RAGDraft.model_json_schema(),
    )
    judge_client = OllamaClient(
        model="gemma3:4b",
        num_gpu=0,
        timeout=300,
        response_format=CoverageAssessment.model_json_schema(),
    )

    judge = EvidenceSufficiencyJudge(
        client=judge_client
    )
    # 5. 调用完整 RAG 流程
    result = guarded_answer_question(
        question=args.question,
        retriever=retriever,
        generator=generator,
        judge=judge,
        top_k=args.top_k,
        max_distance=args.max_distance,
    )

    # 6. 输出回答
    print("\n========== RAG 回答 ==========")

    print("问题：", args.question)
    print("回答状态：", result.status)
    print("回答：", result.answer)

    # 7. 输出程序恢复的真实证据
    print("\n========== 教材出处 ==========")

    if not result.citations:
        print("没有可引用的教材证据")

    for citation in result.citations:
        print(f"\n[{citation.evidence_id}]")
        print("教材：", citation.source)
        print("位置：", citation.location)
        print("原文引用：", citation.quote)


if __name__ == "__main__":
    main()
