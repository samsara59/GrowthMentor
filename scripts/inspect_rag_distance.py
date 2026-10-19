
from pathlib import Path

from app.rag.embedding_client import OllamaEmbeddingClient
from app.rag.vector_store import ChromaKnowledgeStore


ROOT = Path(__file__).resolve().parents[1]

store = ChromaKnowledgeStore(
    db_path=ROOT / "data" / "knowledge.db",
    vector_path=ROOT / "data" / "chroma",
    embedder=OllamaEmbeddingClient(
        model="bge-m3",
        num_gpu=0,
    ),
)

questions = [
    "阅读时，Question是什么意思？",
    "如何用QAS笔记法整理问题、答案和信息出处？",
    "明天天气怎么样？",
]

for question in questions:
    print("\n用户问题：", question)

    hits = store.search(question, limit=3)

    if not hits:
        print("没有候选知识")
        continue

    for hit in hits:
        print(
            "概念：", hit.result.item.concept,
            "余弦距离：", round(hit.distance, 4),
        )
