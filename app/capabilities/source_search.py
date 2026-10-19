
"""从已建立的教材原文集合检索，缺失时显式停用。"""

import json
import os
import urllib.request

from pathlib import Path


COLLECTION = "growthmentor_source_chunks_v1"


# ============================================
# 1. 原始教材检索器
# ============================================

class LocalSourceSearch:

    def __init__(
        self,
        collection,
        model: str = "bge-m3",
        base_url: str | None = None,
    ):

        self.collection = collection

        self.model = model

        self.base_url = (
            base_url
            or os.getenv(
                "GROWTH_OLLAMA_URL",
                "http://127.0.0.1:11434",
            )
        ).rstrip("/")

    # ========================================
    # 2. 把检索问题转换为向量
    # ========================================

    def _embed_openai(self, text: str) -> list[float]:
        """
        OpenAI-compatible embedding endpoint.
        The index must use the same embedding model.
        """
        base_url = os.getenv(
            "EMBEDDING_BASE_URL", ""
        ).strip().rstrip("/")

        model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", ""
        ).strip()

        api_key = os.getenv(
            "EMBEDDING_API_KEY", ""
        )

        if not base_url or not model_name:
            raise ValueError(
                "Set EMBEDDING_BASE_URL and EMBEDDING_MODEL_NAME"
            )

        headers = {
            "Content-Type": "application/json",
        }

        if api_key:
            headers["Authorization"] = (
                "Bearer " + api_key
            )

        payload = {
            "model": model_name,
            "input": text,
        }

        request = urllib.request.Request(
            base_url + "/embeddings",
            data=json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:
            result = json.load(response)

        data = result.get("data")

        if (
            not isinstance(data, list)
            or len(data) != 1
            or not isinstance(data[0], dict)
        ):
            raise RuntimeError(
                "Embedding API returned invalid data"
            )

        vector = data[0].get("embedding")

        if (
            not isinstance(vector, list)
            or not vector
            or any(
                type(value) not in (int, float)
                for value in vector
            )
        ):
            raise RuntimeError(
                "Embedding API returned invalid vector"
            )

        return vector

    def _embed(
        self,
        text: str,
    ) -> list[float]:

        backend = os.getenv(
            "GROWTH_EMBEDDING_BACKEND",
            "ollama",
        ).strip().lower()

        if backend == "openai":
            return self._embed_openai(text)

        if backend != "ollama":
            raise ValueError(
                "Unsupported embedding backend: " + backend
            )

        request = urllib.request.Request(
            self.base_url + "/api/embed",

            data=json.dumps({
                "model": self.model,
                "input": [text],

                "options": {
                    "num_gpu": 0,
                },
            }).encode("utf-8"),

            headers={
                "Content-Type": "application/json"
            },

            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=300,
        ) as response:

            raw = json.load(response)

        vectors = raw.get("embeddings")

        if (
            not isinstance(vectors, list)
            or len(vectors) != 1
        ):

            raise RuntimeError(
                "Embedding返回数量异常"
            )

        return vectors[0]

    # ========================================
    # 3. 通过Chroma检索教材原文
    # ========================================

    def search(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict]:

        if (
            not query.strip()
            or not 1 <= limit <= 10
        ):

            raise ValueError(
                "检索问题或数量无效"
            )

        count = self.collection.count()

        if count == 0:
            return []

        raw = self.collection.query(
            query_embeddings=[
                self._embed(query)
            ],

            n_results=min(
                limit,
                count,
            ),

            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = (
            raw.get("documents") or [[]]
        )[0]

        metadatas = (
            raw.get("metadatas") or [[]]
        )[0]

        distances = (
            raw.get("distances") or [[]]
        )[0]

        results = []

        for doc, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            meta = metadata or {}

            source = meta.get("source")

            if (
                not isinstance(source, str)
                or not source.strip()
                or not isinstance(doc, str)
                or not doc.strip()
            ):
                continue

            results.append({
                "source": source,

                "location": str(
                    meta.get(
                        "location",
                        "not_recorded",
                    )
                ),

                "quote": doc[:500],

                "distance": float(distance),

                "verification": (
                    "retrieved_not_checked"
                ),
            })

        return results


# ============================================
# 4. 将原文检索绑定到现有能力注册表
# ============================================



def verify_retrieved_sources(results, books_dir):
    """
    Check retrieved excerpts against original books.

    Only trusted local verification may set the
    verified_against_original status.
    """
    from app.knowledge.source_provenance import (
        SourceCatalog,
        verify_excerpt,
    )

    if not Path(books_dir).is_dir():
        return results

    catalog = SourceCatalog(books_dir)
    checked = []

    for item in results:
        record = dict(item)

        record["verification"] = "retrieved_not_checked"
        record.pop("original_text", None)

        result = verify_excerpt(
            catalog,
            record.get("source", ""),
            record.get("quote", ""),
        )

        record["provenance_status"] = result.status
        record["provenance_location"] = result.located_at

        if result.status == "matched_original_text":
            # Verification was performed against a real file.
            # Keep only the matched excerpt in session data,
            # not the entire private source document.
            record["verification"] = (
                "verified_against_original"
            )
            record["original_text"] = record["quote"]

        checked.append(record)

    return checked

def connect_source_search(
    registry,
    root: Path,
) -> str:

    folder = (
        Path(root)
        / "data"
        / "chroma"
    )

    if not folder.is_dir():

        return (
            "教材原文Chroma目录不存在；"
            "可先完成第63课索引。"
        )

    try:

        import chromadb

        client = chromadb.PersistentClient(
            path=str(folder)
        )

        # 只打开已存在的Collection
        collection = client.get_collection(
            name=COLLECTION
        )

        if collection.count() == 0:

            return (
                "原文索引存在，但向量数量为0。"
            )

        backend = LocalSourceSearch(
            collection
        )

        books_dir = Path(root) / "books"

        def checked_search(query: str, limit: int = 3):
            results = backend.search(
                query=query,
                limit=limit,
            )

            return verify_retrieved_sources(
                results,
                books_dir,
            )

        registry.bind(
            "knowledge.retrieve",
            checked_search,
        )

        return (
            "已连接原文索引："
            f"{collection.count()}个文本块；"
            "尚未逐字核验来源。"
        )

    except Exception as exc:

        return (
            "未绑定原文检索："
            f"{type(exc).__name__}"
            "（检查Collection和依赖）。"
        )
