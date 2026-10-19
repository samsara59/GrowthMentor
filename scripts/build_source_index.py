
"""从已清洗教材建立独立的原文向量索引。"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from app.knowledge.coverage_audit import (
    filename_from_record,
)


ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = ROOT / "data" / "books_clean.json"
CHROMA_DIR = ROOT / "data" / "chroma"

# 不覆盖旧的结构化知识集合
COLLECTION = "growthmentor_source_chunks_v1"

TEXT_KEYS = (
    "text",
    "clean_text",
    "cleaned_text",
    "content",
    "page_content",
    "page_text",
    "body",
    "paragraph",
    "chunk",
    "extracted_text",
    "raw_text",
)

LOCATION_KEYS = (
    "location",
    "page",
    "page_number",
    "section",
    "chapter",
)


# ============================================
# 1. 加载已有的清洗结果
# ============================================

def load_records(path: Path) -> list:

    content = json.loads(
        path.read_text(encoding="utf-8-sig")
    )

    if isinstance(content, list):
        return content

    if isinstance(content, dict):

        for key in (
            "records",
            "items",
            "documents",
            "chunks",
            "data",
            "pages",
        ):

            value = content.get(key)

            if isinstance(value, list):
                return value

    raise ValueError(
        "books_clean.json的结构不在当前支持范围内，"
        "请根据实际字段适配，不能强行索引。"
    )


# ============================================
# 2. 提取真实文本
# ============================================

def extract_text(record) -> str | None:

    if not isinstance(record, dict):
        return None

    for key in TEXT_KEYS:

        value = record.get(key)

        if isinstance(value, str):
            value = value.strip()

            if value:
                return value

    return None


# ============================================
# 3. 读取原始位置信息
# ============================================

def extract_location(record) -> str:

    for obj in (
        record,
        record.get("metadata"),
    ):

        if not isinstance(obj, dict):
            continue

        for key in LOCATION_KEYS:

            value = obj.get(key)

            if isinstance(value, (str, int)):
                if str(value).strip():
                    return f"{key}={value}"

    return "not_recorded"


# ============================================
# 4. 分块函数
# ============================================

def split_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
):

    if not 0 <= overlap < chunk_size:
        raise ValueError(
            "必须满足 0 <= overlap < chunk_size"
        )

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text),
        )

        yield text[start:end]

        if end == len(text):
            break

        start = end - overlap


# ============================================
# 5. 生成带来源的索引记录
# ============================================

def iter_entries(
    records,
    *,
    max_records=0,
    min_chars=40,
    chunk_size=800,
    overlap=120,
):

    selected = (
        records[:max_records]
        if max_records
        else records
    )

    for index, record in enumerate(selected):

        text = extract_text(record)

        # 复用上节课的来源识别逻辑
        source = filename_from_record(record)

        # 来源或正文缺失时，不编造
        if (
            source is None
            or text is None
            or len(text) < min_chars
        ):
            continue

        location = extract_location(record)

        for chunk_index, chunk in enumerate(
            split_text(
                text,
                chunk_size,
                overlap,
            )
        ):

            if len(chunk.strip()) < min_chars:
                continue

            # 相同输入重复运行得到相同ID
            fingerprint_input = (
                f"{index}:{source}:"
                f"{chunk_index}:{chunk}"
            )

            record_id = hashlib.sha256(
                fingerprint_input.encode("utf-8")
            ).hexdigest()

            metadata = {
                "source": source,
                "location": location,
                "record_index": index,
                "chunk_index": chunk_index,
            }

            yield record_id, chunk, metadata


# ============================================
# 6. 索引之前先检查字段覆盖情况
# ============================================

def preflight(
    records,
    max_records=0,
    min_chars=40,
) -> int:

    selected = (
        records[:max_records]
        if max_records
        else records
    )

    valid = 0
    missing_source = 0
    missing_text = 0
    short_text = 0

    for record in selected:

        source = filename_from_record(record)
        text = extract_text(record)

        if source is None:
            missing_source += 1

        if text is None:
            missing_text += 1

        elif len(text) < min_chars:
            short_text += 1

        if (
            source is not None
            and text is not None
            and len(text) >= min_chars
        ):
            valid += 1

    print("\n========== 教材索引预检 ==========")

    print("清洗记录总数：", len(records))
    print("本次检查数量：", len(selected))
    print("可索引记录：", valid)
    print("缺少可识别来源：", missing_source)
    print("缺少正文：", missing_text)
    print("正文过短：", short_text)

    if selected and isinstance(
        selected[0], dict
    ):

        print(
            "首条记录字段：",
            list(selected[0].keys()),
        )

    print(
        "注意：上述缺失分类可能重叠，"
        "不能简单相加。"
    )

    return valid


# ============================================
# 7. 调用本地Ollama生成Embedding
# ============================================

def embed_texts(
    texts: list[str],
    model: str = "bge-m3",
) -> list[list[float]]:

    payload = json.dumps({
        "model": model,
        "input": texts,
        "options": {
            "num_gpu": 0,
        },
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/embed",
        data=payload,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=300,
    ) as response:

        result = json.load(response)

    vectors = result.get("embeddings")

    if (
        not isinstance(vectors, list)
        or len(vectors) != len(texts)
    ):
        raise RuntimeError(
            "Ollama返回的Embedding数量不正确"
        )

    return vectors


# ============================================
# 8. 将教材写入新的Chroma Collection
# ============================================

def build_index(
    records,
    args,
) -> None:

    import chromadb

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []

    processed = 0

    def flush_batch():

        nonlocal processed

        if not ids:
            return

        vectors = embed_texts(
            documents,
            model=args.model,
        )

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors,
        )

        processed += len(ids)

        print(
            f"本次已写入：{processed} 个文本块，"
            f"集合现有：{collection.count()} 个",
            flush=True,
        )

        ids.clear()
        documents.clear()
        metadatas.clear()

    for record_id, text, metadata in iter_entries(
        records,
        max_records=args.max_records,
        min_chars=args.min_chars,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    ):

        ids.append(record_id)
        documents.append(text)
        metadatas.append(metadata)

        if len(ids) >= args.batch_size:
            flush_batch()

    flush_batch()

    print(
        "\n原文索引写入结束。"
        "旧growthmentor_cosine_v1未被修改。"
    )


# ============================================
# 9. 检索新建立的原文索引
# ============================================

def search_index(
    query: str,
    limit: int,
    model: str,
) -> None:

    import chromadb

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION
    )

    count = collection.count()

    if count == 0:
        print("当前原文索引为空。")
        return

    vectors = embed_texts(
        [query],
        model=model,
    )

    result = collection.query(
        query_embeddings=vectors,
        n_results=min(limit, count),
        include=[
            "metadatas",
            "distances",
        ],
    )

    metadata_list = result["metadatas"][0]
    distance_list = result["distances"][0]

    print("\n========== 原文检索结果 ==========")

    for i, (metadata, distance) in enumerate(
        zip(metadata_list, distance_list),
        start=1,
    ):

        print(
            f"{i}. 文件：{metadata.get('source')}"
        )

        print(
            "   位置：",
            metadata.get("location"),
        )

        print(
            "   距离：",
            round(float(distance), 4),
        )

    print(
        "\n这些仍是检索候选，"
        "尚未经过原文引用与方法适用性核查。"
    )


# ============================================
# 10. Windows命令行入口
# ============================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--build",
        action="store_true",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="bge-m3",
    )

    args = parser.parse_args()

    if (
        args.max_records < 0
        or not 1 <= args.batch_size <= 64
        or args.min_chars < 1
        or not 0 <= args.overlap < args.chunk_size
        or not 1 <= args.limit <= 100
    ):
        parser.error(
            "检索、分块或批处理参数不合法"
        )

    # 仅检索已经建立的原文索引
    if (
        args.query is not None
        and not args.build
        and not args.dry_run
    ):

        search_index(
            args.query,
            args.limit,
            args.model,
        )

        return

    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(
            f"找不到清洗文件：{SOURCE_FILE}"
        )

    records = load_records(
        SOURCE_FILE
    )

    count = preflight(
        records,
        max_records=args.max_records,
        min_chars=args.min_chars,
    )

    if args.build:

        if count == 0:
            raise RuntimeError(
                "没有可索引记录，"
                "已阻止写入空索引。"
            )

        build_index(
            records,
            args,
        )

    else:

        print(
            "\n预检完成，没有写入Chroma。"
            "必须明确使用--build才能建立索引。"
        )


if __name__ == "__main__":
    main()
