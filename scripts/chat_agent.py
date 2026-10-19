
import argparse
from pathlib import Path

from app.agent.schema import RouteDecision
from app.agent.router import LLMAgentRouter
from app.agent.tools import RAGKnowledgeTool

from app.agent.orchestrator import AgentOrchestrator

from app.agent.guidance import (
    render_choices,
    explain_choice,
)

from app.assessment.schema import AssessmentDraft
from app.assessment.assessor import ExperienceAssessor

from app.assessment.cross_schema import (
    CrossAssessmentDraft,
)

from app.assessment.cross_assessor import (
    CrossExperienceAssessor,
)

from app.assessment.verification import (
    VerificationJudgment,
    EvidenceVerifier,
)

from app.knowledge.ollama_client import OllamaClient

from app.rag.embedding_client import OllamaEmbeddingClient
from app.rag.vector_store import ChromaKnowledgeStore
from app.rag.pipeline import RAGDraft

from app.rag.answerability import (
    CoverageAssessment,
    EvidenceSufficiencyJudge,
)

from app.memory.schema import AgentSession
from app.memory.store import MemoryStore


ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_DB = ROOT / "data" / "knowledge.db"
VECTOR_PATH = ROOT / "data" / "chroma"
MEMORY_DB = ROOT / "data" / "user_memory.db"


# ============================================
# 1. 初始化所有模块
# ============================================

def build_agent() -> AgentOrchestrator:

    # A. Embedding
    embedder = OllamaEmbeddingClient(
        model="bge-m3",
        num_gpu=0,
        timeout=300,
    )

    # B. 教材向量检索
    retriever = ChromaKnowledgeStore(
        db_path=KNOWLEDGE_DB,
        vector_path=VECTOR_PATH,
        embedder=embedder,
    )

    # C. RAG证据充分性判断
    judge_client = OllamaClient(
        model="gemma3:4b",
        num_gpu=0,
        timeout=300,
        response_format=(
            CoverageAssessment.model_json_schema()
        ),
    )

    judge = EvidenceSufficiencyJudge(
        client=judge_client
    )

    # D. RAG回答生成
    answer_client = OllamaClient(
        model="gemma3:4b",
        num_gpu=0,
        timeout=300,
        response_format=RAGDraft.model_json_schema(),
    )

    knowledge_tool = RAGKnowledgeTool(
        retriever=retriever,
        generator=answer_client,
        judge=judge,
    )

    # E. Agent意图路由
    router_client = OllamaClient(
        model="gemma3:4b",
        num_gpu=0,
        timeout=300,
        response_format=RouteDecision.model_json_schema(),
    )

    router = LLMAgentRouter(
        client=router_client
    )

    # F. 单经历Assessment
    assessment_client = OllamaClient(
        model="gemma3:4b",
        num_gpu=0,
        timeout=300,
        response_format=AssessmentDraft.model_json_schema(),
    )

    assessor = ExperienceAssessor(
        client=assessment_client,
        max_attempts=2,
    )

    # G. 跨经历Assessment
    cross_client = OllamaClient(
        model="gemma3:4b",
        num_gpu=0,
        timeout=300,
        response_format=CrossAssessmentDraft.model_json_schema(),
    )

    cross_assessor = CrossExperienceAssessor(
        client=cross_client,
        max_attempts=2,
    )

    # H. 本课新增的动态验证模块
    verification_client = OllamaClient(
        model="gemma3:4b",
        num_gpu=0,
        timeout=300,
        response_format=(
            VerificationJudgment.model_json_schema()
        ),
    )

    verifier = EvidenceVerifier(
        client=verification_client,
        max_attempts=2,
    )

    # I. 组装完整Agent
    return AgentOrchestrator(
        router=router,
        knowledge_tool=knowledge_tool,
        assessor=assessor,
        cross_assessor=cross_assessor,
        verifier=verifier,
    )


# ============================================
# 2. 显示所有命令
# ============================================

def print_help() -> None:

    print("\n可用命令：")

    print("  /start                 新的经历访谈")
    print("  /ask 问题              教材问答")
    print("  /compare               跨经历综合分析")
    print("  /verify 1              验证报告中的第1个候选方向")
    print("  /retryverify           重试已暂存的验证回答")
    print("  /verification          查看验证记录")
    print("  /cancelverify          结束进行中的验证")
    print("  /history               查看历史经历")
    print("  /status                查看会话状态")
    print("  /reanalyze 1           重新分析第1条经历")
    print("  /save                  手动保存")
    print("  /edit 1 action 内容    更正第1条经历的行动")
    print("  /delete 1             删除第1条经历")
    print("  /reset                 放弃当前访谈/验证草稿")
    print("  /forget                清除当前会话记忆")
    print("  /help                  显示帮助")
    print("  exit                   退出程序")


# ============================================
# 3. 查看历史经历
# ============================================

def print_history(
    session: AgentSession,
) -> None:

    print("\n========== 历史经历 ==========")

    if not session.experiences:
        print("没有完整经历。")
        return

    for i, record in enumerate(
        session.experiences,
        1,
    ):

        print(f"\n【经历 {i}】")
        print("情境：", record.situation)
        print("行动：", record.action)
        print("结果：", record.outcome)

        assessment = session.assessments[i - 1]

        if assessment is None:
            print(
                "分析状态：尚未完成或已失效"
                f"（可输入 /reanalyze {i}）"
            )

        else:
            print(
                "分析状态：已保存单经历分析"
            )

            for candidate in assessment.hypotheses:
                print(
                    "  候选方向：",
                    candidate.name,
                )

            print(
                "  下一轮建议问题：",
                assessment.follow_up_question,
            )


# ============================================
# 4. 查看假设验证过程
# ============================================

def print_verification(
    session: AgentSession,
) -> None:

    state = session.verification

    print("\n========== 假设验证记录 ==========")

    if state is None:
        print(
            "没有当前验证记录。"
            "请先执行 /compare，再输入 /verify 1。"
        )
        return

    print("候选方向：", state.candidate_name)

    print(
        "已完成轮数：",
        len(state.turns),
    )

    print(
        "状态：",
        "已完成" if state.completed else "进行中",
    )

    meanings = {
        "supports": "支持线索",
        "challenges": "可能的限制线索",
        "unclear": "尚不明确",
    }

    for i, turn in enumerate(
        state.turns,
        1,
    ):

        print(
            f"\n第{i}轮问题：{turn.question}"
        )

        print(
            "用户回答：",
            turn.answer,
        )

        print(
            "判断：",
            meanings[turn.relation],
        )

        print(
            "原文引用：",
            turn.quote or "无",
        )

        print(
            "解释：",
            turn.reason,
        )

    # 模型失败，但用户回答已经暂存
    if state.pending_answer is not None:

        print(
            "\n尚未完成分析的回答：",
            state.pending_answer,
        )

        print(
            "可输入 /retryverify 重试。"
        )

    elif session.stage == "awaiting_verification":

        print(
            "\n下一轮问题：",
            state.pending_question,
        )


# ============================================
# 5. 统一显示Agent回答
# ============================================

def print_reply(reply) -> None:

    print("\nAgent：\n" + reply.message)

    print(
        "\n[处理模块]",
        reply.mode,
    )

    if reply.status is not None:
        print(
            "回答状态：",
            reply.status,
        )

    for citation in reply.citations:

        print(
            f"[{citation.evidence_id}] "
            f"{citation.source} "
            f"{citation.location}"
        )

        print(
            "原文：",
            citation.quote,
        )


# ============================================
# 6. 主程序
# ============================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--session",
        type=str,
        default="local_default",
    )

    session_id = parser.parse_args().session

    # 从本地数据库恢复状态
    memory = MemoryStore(MEMORY_DB)

    session = memory.load_session(
        session_id
    )

    agent = build_agent()

    print("=" * 48)
    print("GrowthMentor-Agent — Dynamic Verification")
    print("=" * 48)

    print("会话：", session_id)
    print("已恢复经历：", len(session.experiences))
    print("当前阶段：", session.stage)

    print_help()

    # ========================================
    # 7. 恢复上次未完成的任务
    # ========================================

    if (
        session.stage == "awaiting_verification"
        and session.verification is not None
    ):

        print(
            "\nAgent：已恢复上一轮假设验证。"
        )

        if session.verification.pending_answer is not None:
            print(
                "上次回答已经保存，但分析失败。"
                "请输入 /retryverify。"
            )

        else:
            print(
                "请继续回答：",
                session.verification.pending_question,
            )

    elif session.stage != "idle":

        print(
            "\nAgent：已恢复未完成的经历访谈，"
            "请继续回答当前阶段。"
        )

        print(
            render_choices(session.stage)
        )

    pending_delete = None

    # ========================================
    # 8. 主聊天循环
    # ========================================

    while True:

        try:
            message = input("\n你：").strip()

        except (EOFError, KeyboardInterrupt):
            print("\n聊天结束。")
            break

        if message.lower() in {"exit", "quit"}:
            print("聊天结束。")
            break

        if not message:
            continue

        # ------------------------------------
        # 9. 处理删除确认
        # ------------------------------------

        if pending_delete is not None:

            operation, number = pending_delete
            pending_delete = None

            if message != "YES":
                print("已取消删除。")
                continue

            try:
                if operation == "experience":

                    session = memory.delete_experience(
                        session_id,
                        session,
                        number,
                    )

                    print(
                        f"已删除经历{number}及关联分析，"
                        "旧跨经历报告和验证记录已失效。"
                    )

                else:

                    memory.forget_session(
                        session_id
                    )

                    session = AgentSession()

                    print(
                        "已清除当前会话在SQLite中的记录。"
                    )

            except Exception as exc:
                print(
                    "删除失败：",
                    exc,
                )

            continue

        # ------------------------------------
        # 10. 帮助
        # ------------------------------------

        if message == "/help":
            print_help()
            continue

        # ------------------------------------
        # 11. 会话状态
        # ------------------------------------

        if message == "/status":

            print("\n========== 会话状态 ==========")

            print("会话：", session_id)
            print("阶段：", session.stage)

            print(
                "已完成经历数：",
                len(session.experiences),
            )

            print(
                "已完成初步分析数：",
                sum(
                    item is not None
                    for item in session.assessments
                ),
            )

            print(
                "已缓存跨经历报告：",
                session.cross_report is not None,
            )

            print(
                "已完成验证轮数：",
                len(session.verification.turns)
                if session.verification
                else 0,
            )

            print(
                "当前草稿字段：",
                list(session.draft),
            )

            if (
                session.stage == "awaiting_verification"
                and session.verification is not None
            ):

                print(
                    "验证方向：",
                    session.verification.candidate_name,
                )

                print(
                    "当前问题：",
                    session.verification.pending_question,
                )

            elif session.stage != "idle":

                print(
                    render_choices(session.stage)
                )

            continue

        # ------------------------------------
        # 12. 查看历史经历
        # ------------------------------------

        if message == "/history":
            print_history(session)
            continue

        # ------------------------------------
        # 13. 查看验证记录
        # ------------------------------------

        if message == "/verification":
            print_verification(session)
            continue

        # ------------------------------------
        # 14. 手动保存
        # ------------------------------------

        if message == "/save":

            try:
                memory.save_session(
                    session_id,
                    session,
                )

                print(
                    "已保存到本地SQLite。经历数：",
                    len(session.experiences),
                )

            except Exception as exc:
                print(
                    "保存失败：",
                    exc,
                )

            continue

        # ------------------------------------
        # 15. 重新分析历史经历
        # ------------------------------------

        if message.startswith("/reanalyze"):

            parts = message.split()

            if (
                len(parts) != 2
                or parts[0] != "/reanalyze"
            ):
                print(
                    "用法：/reanalyze 1"
                )
                continue

            try:
                reply = agent.reanalyze_experience(
                    int(parts[1]),
                    session,
                )

                memory.save_session(
                    session_id,
                    session,
                )

                print_reply(reply)

            except Exception as exc:
                print(
                    "重新分析失败：",
                    exc,
                )

            continue

        # ------------------------------------
        # 16. 更正历史经历
        # ------------------------------------

        if message.startswith("/edit "):

            parts = message.split(
                maxsplit=3
            )

            if len(parts) != 4:
                print(
                    "用法：/edit 1 action 更正内容"
                )
                continue

            try:
                session = memory.update_experience(
                    session_id,
                    session,
                    int(parts[1]),
                    parts[2],
                    parts[3],
                )

                print(
                    "已更正经历。"
                    "旧单经历分析、跨经历报告"
                    "和验证记录均已失效。"
                )

            except Exception as exc:
                print(
                    "更正失败：",
                    exc,
                )

            continue

        # ------------------------------------
        # 17. 删除指定经历
        # ------------------------------------

        if message.startswith("/delete "):

            parts = message.split()

            if len(parts) != 2:
                print(
                    "用法：/delete 1"
                )
                continue

            try:
                number = int(parts[1])

                if not 1 <= number <= len(
                    session.experiences
                ):
                    raise ValueError(
                        "经历编号不存在"
                    )

                pending_delete = (
                    "experience",
                    number,
                )

                print(
                    f"即将删除经历{number}及相关分析。"
                    "输入 YES 确认，其他内容取消。"
                )

            except ValueError as exc:
                print(
                    "错误：",
                    exc,
                )

            continue

        # ------------------------------------
        # 18. 删除当前会话记忆
        # ------------------------------------

        if message == "/forget":

            pending_delete = ("all", None)

            print(
                "这将删除当前会话的所有记录。"
                "输入 YES 确认，其他内容取消。"
            )

            continue

        # ------------------------------------
        # 19. 重置未完成流程
        # ------------------------------------

        if message == "/reset":

            session.stage = "idle"
            session.draft.clear()

            session.verification = None

            try:
                memory.save_session(
                    session_id,
                    session,
                )

                print(
                    "已放弃当前草稿，"
                    "已完成经历和跨经历报告仍保留。"
                )

            except Exception as exc:
                print(
                    "保存失败：",
                    exc,
                )

            continue

        # ------------------------------------
        # 20. 数字选项引导
        # ------------------------------------

        hint = explain_choice(
            session.stage,
            message,
        )

        if hint is not None:
            print(
                "\nAgent：",
                hint,
            )
            continue

        # ------------------------------------
        # 21. 正式调用Agent
        # ------------------------------------

        try:
            reply = agent.handle(
                message,
                session,
            )

        except Exception as exc:

            print(
                "Agent执行异常：",
                type(exc).__name__,
                str(exc),
            )

            try:
                memory.save_session(
                    session_id,
                    session,
                )

            except Exception as save_exc:
                print(
                    "警告：会话未保存：",
                    save_exc,
                )

            continue

        # ------------------------------------
        # 22. 自动保存
        # ------------------------------------

        try:
            memory.save_session(
                session_id,
                session,
            )

        except Exception as exc:

            print_reply(reply)

            print(
                "警告：本轮状态未保存：",
                str(exc),
            )

            continue

        # ------------------------------------
        # 23. 打印结果
        # ------------------------------------

        print_reply(reply)

        print(
            "[本地自动保存] 会话已写入SQLite"
        )

        # 普通经历访谈继续提供选项
        if session.stage in {
            "awaiting_situation",
            "awaiting_action",
            "awaiting_outcome",
        }:
            print(
                render_choices(session.stage)
            )


if __name__ == "__main__":
    main()
