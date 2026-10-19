
import json
import hashlib

from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]

input_path = ROOT / "data" / "books_raw.json"

records = json.loads(
    input_path.read_text(encoding="utf-8")
)

# 1. 按教材文件名分组
documents = defaultdict(list)

for record in records:
    documents[record["source"]].append(
        record["text"]
    )

# 2. 计算每份教材的正文哈希
hash_groups = defaultdict(list)

for filename, paragraphs in documents.items():

    # 保留原有文本与记录顺序
    content = "\n".join(paragraphs)

    if not content.strip():
        continue

    digest = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    hash_groups[digest].append(filename)

# 3. 输出重复组
duplicate_groups = [
    filenames
    for filenames in hash_groups.values()
    if len(filenames) > 1
]

print("有文本的教材数：", len(documents))
print("候选重复组数：", len(duplicate_groups))

for index, group in enumerate(
    duplicate_groups, start=1
):
    print(f"\n重复组 {index}")

    for filename in group:
        print("  ", filename)
