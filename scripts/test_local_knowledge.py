
from app.ingestion.schema import DocumentRecord

from app.knowledge.llm_extractor import (
    LLMExtractor,
    KnowledgeBatch,
)

from app.knowledge.ollama_client import OllamaClient

from app.knowledge.extractors import run_extractor


class DebugClient:
    """打印大模型原始输出的调试客户端。"""

    def __init__(self, client):
        self.client = client

    def generate(self, prompt: str) -> str:

        response = self.client.generate(prompt)

        print("\n=== 模型原始输出 ===")
        print(response)

        return response


def main():

    # =============================================
    # 1. 构造人工测试文本
    # =============================================

    record = DocumentRecord(
        source="demo.pdf",
        location="第1页",
        text=(
            "学习力是指个体获取、理解和运用知识的能力。\n"
            "复盘是指对过去行动进行回顾并总结经验的过程。"
        ),
        method="pdf_text",
    )

    # =============================================
    # 2. 获取知识结构的 JSON Schema
    # =============================================

    knowledge_schema = (
        KnowledgeBatch.model_json_schema()
    )

    # =============================================
    # 3. 创建 Ollama 客户端
    # =============================================

    client = OllamaClient(
        model="gemma3:4b",

        # 当前 GPU CUDA 存在兼容问题
        num_gpu=0,

        timeout=300,

        # 使用 Pydantic Schema 约束模型输出
        response_format=knowledge_schema,
    )

    # =============================================
    # 4. 创建知识提取器
    # =============================================

    extractor = LLMExtractor(
        client=DebugClient(client),
        method="local_llm",
    )

    # =============================================
    # 5. 调用模型并核验原文
    # =============================================

    try:

        accepted, rejected = run_extractor(
            extractor,
            record,
        )

    except ValueError as exc:

        print("\n=== 结构验证失败 ===")
        print(exc)
        return

    # =============================================
    # 6. 输出核验统计
    # =============================================

    print("\n=== 核验统计 ===")

    print("通过数量：", len(accepted))
    print("拒绝数量：", len(rejected))

    # =============================================
    # 7. 当前测试的知识覆盖检查
    # =============================================

    expected_concepts = {
        "学习力",
        "复盘"
    }

    extracted_concepts = {
        item.concept
        for item in accepted
    }

    missing_concepts = (
        expected_concepts - extracted_concepts
    )

    print("\n=== 知识覆盖检查 ===")
    print("预期概念：", expected_concepts)
    print("实际提取：", extracted_concepts)
    print("遗漏概念：", missing_concepts)

    # =============================================
    # 8. 打印通过核验的知识
    # =============================================

    for item in accepted:

        print("\n=== 通过原文核验的知识 ===")

        print(
            item.model_dump_json(indent=2)
        )

    # =============================================
    # 9. 打印未通过核验的知识
    # =============================================

    for item in rejected:

        print("\n=== 未通过原文核验的知识 ===")

        print(
            item.model_dump_json(indent=2)
        )


if __name__ == "__main__":
    main()
