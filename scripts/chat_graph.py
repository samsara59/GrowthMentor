
import argparse
from pathlib import Path

from app.agent.graph_workflow import (
    build_workflow,
)

from app.agent.guidance import (
    explain_choice,
    render_choices,
)

from app.memory.store import MemoryStore

# 复用旧程序已经调试完成的模块初始化
from scripts.chat_agent import build_agent


ROOT = Path(__file__).resolve().parents[1]
MEMORY_DB = ROOT / "data" / "user_memory.db"


def print_reply(reply):
    """显示统一的Agent回复。"""

    print("\nAgent：")
    print(reply.message)

    print("\n[处理模块]", reply.mode)

    if reply.status is not None:
        print("回答状态：", reply.status)

    for citation in reply.citations:
        print(
            f"[{citation.evidence_id}] "
            f"{citation.source} "
            f"{citation.location}"
        )
        print("原文：", citation.quote)


def print_commands():
    print("\n可用命令：")
    print("  /start               开始优势探索")
    print("  /ask 问题            教材知识问答")
    print("  /compare             跨经历分析")
    print("  /verify 1            开始假设验证")
    print("  /retryverify         重试验证")
    print("  /cancelverify        结束验证")
    print("  /verification        查看验证记录")
    print("  /reanalyze 1         重新分析经历")
    print("  /history             查看历史经历")
    print("  /status              查看会话状态")
    print("  /save                手动保存")
    print("  /edit 1 action 内容  更正经历")
    print("  /delete 1           请求删除经历")
    print("  /forget             请求清除记忆")
    print("  /reset              放弃当前未完成的任务")
    print("  exit                退出")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--session",
        default="local_default",
        type=str,
    )

    parser.add_argument(
        "--draw-graph",
        action="store_true",
        help="导出LangGraph的Mermaid结构，不启动LLM",
    )

    args = parser.parse_args()

    # ========================================
    # A. 只导出图结构
    # ========================================

    if args.draw_graph:

        graph = build_workflow()

        diagram = graph.get_graph().draw_mermaid()

        output = ROOT / "data" / "agent_graph.mmd"

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            diagram,
            encoding="utf-8",
        )

        print("图结构已保存：", output)
        print("\n" + diagram)

        return

    # ========================================
    # B. 初始化正式Agent与Memory
    # ========================================

    memory = MemoryStore(MEMORY_DB)

    agent = build_agent()

    graph = build_workflow(
        agent=agent,
        memory=memory,
    )

    session_id = args.session

    session = memory.load_session(session_id)

    print("=" * 52)
    print("GrowthMentor-Agent — LangGraph")
    print("=" * 52)

    print("会话：", session_id)
    print("已恢复经历：", len(session.experiences))
    print("当前阶段：", session.stage)

    print_commands()

    # ========================================
    # C. 恢复上一次任务
    # ========================================

    if session.stage == "awaiting_verification":

        verification = session.verification

        if verification is not None:

            if verification.pending_answer:
                print(
                    "\n上次回答已保存，"
                    "但分析尚未完成。"
                    "请使用 /retryverify。"
                )

            else:
                print(
                    "\n继续回答：",
                    verification.pending_question,
                )

    elif session.stage != "idle":

        print(
            "\n已恢复尚未完成的经历访谈。"
        )

        print(
            render_choices(session.stage)
        )

    # ========================================
    # D. 交互循环
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

        # 保留第52课的数字选项引导
        current = memory.load_session(
            session_id
        )

        hint = explain_choice(
            current.stage,
            message,
        )

        if hint is not None:
            print("\nAgent：", hint)
            continue

        # ====================================
        # E. 运行一轮LangGraph
        # ====================================

        try:
            result = graph.invoke(
                {
                    "message": message,
                    "session_id": session_id,
                }
            )

        except Exception as exc:

            print(
                "\n工作流执行失败：",
                type(exc).__name__,
                str(exc),
            )

            continue

        # ====================================
        # F. 显示结果与执行轨迹
        # ====================================

        print_reply(
            result["reply"]
        )

        print(
            "\n[LangGraph执行路径]"
        )

        print(
            " → ".join(result["path"])
        )

        if result.get("skip_save"):
            print(
                "[Memory] 已执行会话删除，"
                "没有重新创建记录。"
            )
        else:
            print(
                "[Memory] 本轮会话已写入SQLite。"
            )

        # ====================================
        # G. 下一轮引导
        # ====================================

        updated = result["session"]

        if updated.stage in {
            "awaiting_situation",
            "awaiting_action",
            "awaiting_outcome",
        }:
            print(
                render_choices(updated.stage)
            )


if __name__ == "__main__":
    main()
