
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator, Protocol


# ============================================
# 1. Supervisor路由与节点的对应关系
# ============================================

ROUTE_TO_NODE = {
    "rag": "rag_agent",
    "interview": "interview_agent",
    "assessment": "assessment_agent",
    "verification": "verification_agent",
    "memory": "memory_agent",
    "other": "unsupported",
}

WORK_NODES = frozenset(
    ROUTE_TO_NODE.values()
)


# ============================================
# 2. LangGraph流式接口
# ============================================

class GraphLike(Protocol):

    def stream(
        self,
        inputs: dict,
        *,
        stream_mode: str,
    ) -> Iterator[dict]:
        ...


# ============================================
# 3. 单个节点完成后的进度记录
# ============================================

@dataclass(frozen=True)
class NodeProgress:

    node: str
    path: tuple[str, ...]

    route: str | None
    reply: Any | None

    saved: bool
    elapsed_seconds: float


# ============================================
# 4. 预测下一节点
# ============================================

def next_expected_node(
    last_node: str,
    route: str | None,
) -> str | None:

    if last_node == "memory_load":
        return "supervisor"

    if last_node == "supervisor":
        return ROUTE_TO_NODE.get(route)

    if last_node in WORK_NODES:
        return "memory_save"

    return None


# ============================================
# 5. 监听一次完整工作流
# ============================================

def stream_turn(
    graph: GraphLike,
    message: str,
    session_id: str,
) -> Iterator[NodeProgress]:

    if not message.strip():
        raise ValueError(
            "用户消息不能为空"
        )

    if not session_id.strip():
        raise ValueError(
            "会话ID不能为空"
        )

    route = None
    reply = None

    visited: list[str] = []

    saved = False
    start = perf_counter()

    valid_nodes = {
        "memory_load",
        "supervisor",
        "memory_save",
    } | WORK_NODES

    # ========================================
    # 6. 从LangGraph逐个读取完成事件
    # ========================================

    for chunk in graph.stream(
        {
            "message": message,
            "session_id": session_id,
        },
        stream_mode="updates",
    ):

        if not isinstance(chunk, dict):
            raise TypeError(
                "无法识别的LangGraph事件格式"
            )

        for node, changes in chunk.items():

            # 跳过框架保留事件
            if node.startswith("__"):
                continue

            if node not in valid_nodes:
                raise ValueError(
                    f"未知LangGraph节点：{node}"
                )

            if not isinstance(changes, dict):
                raise TypeError(
                    f"节点{node}返回的更新不是字典"
                )

            # =================================
            # 7. 获取当前路由
            # =================================

            if "route" in changes:
                route = changes["route"]

            # =================================
            # 8. 记录业务回复
            # =================================

            if "reply" in changes:
                reply = changes["reply"]

            # =================================
            # 9. 更新已完成的节点列表
            # =================================

            node_path = changes.get("path")

            if (
                isinstance(node_path, list)
                and all(
                    isinstance(item, str)
                    for item in node_path
                )
            ):
                visited = node_path.copy()

            else:
                visited.append(node)

            # =================================
            # 10. 判断持久化节点是否完成
            # =================================

            saved = node == "memory_save"

            # =================================
            # 11. 向网页发送当前进度
            # =================================

            yield NodeProgress(
                node=node,
                path=tuple(visited),
                route=route,
                reply=reply,
                saved=saved,
                elapsed_seconds=(
                    perf_counter() - start
                ),
            )

    # ========================================
    # 12. 工作流结束后检查完整性
    # ========================================

    if not saved:
        raise RuntimeError(
            "工作流没有完成Memory Save，"
            "不能报告保存成功"
        )

    if reply is None:
        raise RuntimeError(
            "工作流完成但没有产生Agent回复"
        )
