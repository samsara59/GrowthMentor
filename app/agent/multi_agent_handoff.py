
"""第76课：带消息协议、证据闸门与有限重试的多Agent图。"""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)
from langgraph.types import interrupt

from app.agent.supervisor_graph import (
    SupervisorPlan,
    TaskSpec,
)
from app.agent.experiment_protocol import (
    render_protocol,
)
from app.knowledge.source_provenance import (
    SourceCatalog,
    verify_excerpt,
)


# ============================================
# 1. Agent之间的消息协议
# ============================================

class Handoff(BaseModel):
    """
    跨Agent传递的数据契约。

    不允许传递数据库连接、
    Python任意对象或未定义字段。
    """

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(min_length=3)
    task_id: str = Field(min_length=2)

    sender: Literal[
        "knowledge",
        "evidence",
        "practice",
    ]

    recipient: Literal[
        "evidence",
        "supervisor",
    ]

    status: Literal[
        "ok",
        "empty",
        "unavailable",
        "proposed",
        "error",
    ]

    attempt: int = Field(ge=1, le=2)

    payload: dict = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def check_route(self):

        valid = {
            ("knowledge", "evidence"),
            ("evidence", "supervisor"),
            ("practice", "supervisor"),
        }

        if (
            self.sender,
            self.recipient,
        ) not in valid:

            raise ValueError(
                "非法Agent交接路线"
            )

        return self


# ============================================
# 2. Knowledge Agent
# ============================================

class KnowledgeAgent:
    """
    只接收知识检索任务，
    不读取用户全部长期记忆。
    """

    def __init__(self, registry):
        self.registry = registry

    def execute(
        self,
        task: TaskSpec,
        attempt: int,
    ) -> Handoff:

        evidence = []
        status = "unavailable"
        error = ""

        if self.registry.is_bound(
            "knowledge.retrieve"
        ):

            try:

                hits = self.registry.invoke(
                    "knowledge.retrieve",
                    query=task.query,
                    limit=3,
                )

                if not isinstance(
                    hits,
                    list,
                ):

                    raise TypeError(
                        "knowledge.retrieve必须返回list"
                    )

                for hit in hits[:3]:

                    if (
                        not isinstance(hit, dict)
                        or not hit.get("source")
                        or not hit.get("quote")
                    ):
                        continue

                    evidence.append({
                        "evidence_id": (
                            f"{task.task_id.upper()}"
                            f"E{len(evidence) + 1}"
                        ),

                        "source": str(
                            hit["source"]
                        )[:240],

                        "location": str(
                            hit.get(
                                "location",
                                "not_recorded",
                            )
                        )[:120],

                        "quote": str(
                            hit["quote"]
                        )[:650],

                        "verification": (
                            "retrieved_not_checked"
                        ),
                    })

                status = (
                    "ok"
                    if evidence
                    else "empty"
                )

            except Exception as exc:

                status = "error"

                error = (
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:150]}"
                )

        return Handoff(
            message_id=(
                f"{task.task_id}:knowledge:{attempt}"
            ),

            task_id=task.task_id,
            sender="knowledge",
            recipient="evidence",

            attempt=attempt,
            status=status,

            payload={
                "evidence": evidence,
                "error": error,
            },
        )


# ============================================
# 3. Evidence Verification Agent
# ============================================

class EvidenceAgent:
    """
    负责原文字符匹配。

    注意：
    字符匹配不是语义蕴涵判断。
    """

    def __init__(
        self,
        catalog: SourceCatalog,
    ):
        self.catalog = catalog

    def execute(
        self,
        message: Handoff,
    ) -> Handoff:

        cleaned = []

        for hit in message.payload.get(
            "evidence",
            [],
        ):

            item = dict(hit)

            verdict = verify_excerpt(
                self.catalog,
                item["source"],
                item["quote"],
            )

            item["verification"] = (
                verdict.status
            )

            item["original_location"] = (
                verdict.located_at
                or "not_located"
            )

            item["original_path"] = (
                verdict.resolved_path
            )

            cleaned.append(item)

        return Handoff(
            message_id=(
                f"{message.task_id}:evidence:"
                f"{message.attempt}"
            ),

            task_id=message.task_id,

            sender="evidence",
            recipient="supervisor",

            attempt=message.attempt,
            status=message.status,

            payload={
                "evidence": cleaned,

                "error": message.payload.get(
                    "error",
                    "",
                ),
            },
        )


# ============================================
# 4. Practice Agent
# ============================================

class PracticeAgent:
    """
    只设计尚待确认的实践。

    不自动写入PracticeStore，
    不替用户确认或执行。
    """

    def __init__(self, designer):
        self.designer = designer

    def execute(
        self,
        task: TaskSpec,
        goal: str,
        attempt: int,
    ) -> Handoff:

        try:

            plan = self.designer.build(
                goal=goal,

                history=[],

                draft=task.objective,

                options=(
                    task.comparison_options
                ),
            )

            status = "proposed"

            payload = {
                "protocol": (
                    plan.model_dump()
                ),

                "preview": (
                    render_protocol(plan)
                ),
            }

        except Exception as exc:

            status = "error"

            payload = {
                "error": (
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:150]}"
                ),
            }

        return Handoff(
            message_id=(
                f"{task.task_id}:practice:{attempt}"
            ),

            task_id=task.task_id,

            sender="practice",
            recipient="supervisor",

            attempt=attempt,
            status=status,

            payload=payload,
        )


# ============================================
# 5. Coaching Agent：只接收必要交付结果
# ============================================

class CoachingAgent:
    """
    在综合回复时，只把完成原文字符匹配
    的候选片段作为可引用的教材文本。

    保留原始Worker结果供UI审计，
    不将未核验的全文交给Coaching。
    """

    def __init__(self, synthesizer):
        self.synthesizer = synthesizer

    def execute(
        self,
        question: str,
        goal: str,
        results: list[dict],
    ):

        safe = []

        for result in results:

            if result["worker"] == "knowledge":

                accepted = [
                    hit

                    for hit in result.get(
                        "evidence",
                        [],
                    )

                    if hit["verification"]
                    == "matched_original_text"
                ]

                safe.append({
                    "task_id": result["task_id"],

                    "worker": "knowledge",

                    "status": result["status"],

                    "evidence": accepted,

                    "unverified_count": (
                        len(
                            result.get(
                                "evidence",
                                [],
                            )
                        )
                        - len(accepted)
                    ),
                })

            else:

                safe.append({
                    "task_id": result["task_id"],

                    "worker": "practice",

                    "status": result["status"],

                    "protocol": result.get(
                        "protocol"
                    ),

                    "error": result.get(
                        "error",
                        "",
                    ),
                })

        reply = self.synthesizer.answer(
            question,
            goal,
            safe,
        )

        allowed = {
            hit["evidence_id"]

            for entry in safe

            for hit in entry.get(
                "evidence",
                [],
            )
        }

        if not set(
            reply.evidence_ids
        ).issubset(allowed):

            raise ValueError(
                "Coaching引用未通过"
                "原文字符匹配的证据"
            )

        return reply


# ============================================
# 6. LangGraph共享状态
# ============================================

class AgentState(
    TypedDict,
    total=False,
):

    question: str
    goal: str
    confirmed_notes: list[dict]

    plan: dict
    approved: bool

    cursor: int
    attempts: dict[str, int]

    pending: dict
    handoffs: list[dict]

    worker_results: list[dict]

    retry_current: bool

    phase: str

    answer: str
    evidence_ids: list[str]
    limitations: str


# ============================================
# 7. 构建完整Multi-Agent工作流
# ============================================

def build_multi_agent_graph(
    planner,
    registry,
    designer,
    coaching,
    catalog,
    checkpointer,
):

    knowledge_agent = KnowledgeAgent(
        registry
    )

    evidence_agent = EvidenceAgent(
        catalog
    )

    practice_agent = PracticeAgent(
        designer
    )

    coaching_agent = CoachingAgent(
        coaching
    )

    # ----------------------------------------
    # A. Supervisor规划
    # ----------------------------------------

    def supervisor(state: AgentState):

        if (
            not str(
                state.get("question", "")
            ).strip()
            or not str(
                state.get("goal", "")
            ).strip()
        ):

            raise ValueError(
                "问题和成长目标不能为空"
            )

        plan = planner.plan(
            state["question"],
            state["goal"],

            state.get(
                "confirmed_notes",
                [],
            ),
        )

        return {
            "plan": (
                SupervisorPlan.model_validate(
                    plan
                ).model_dump()
            ),

            "approved": False,
            "cursor": 0,

            "attempts": {},
            "pending": {},
            "handoffs": [],
            "worker_results": [],

            "retry_current": False,

            "phase": "awaiting_approval",

            "answer": "",
            "evidence_ids": [],
            "limitations": "",
        }

    # ----------------------------------------
    # B. 用户审批
    # ----------------------------------------

    def approval(state: AgentState):

        agreed = interrupt({
            "type": "approve_plan",
            "plan": state["plan"],
        })

        return {
            "approved": agreed is True,

            "phase": (
                "approved"
                if agreed is True
                else "cancelled"
            ),
        }

    def after_approval(
        state: AgentState,
    ):

        return (
            "router"
            if state.get("approved")
            else "cancel"
        )

    # ----------------------------------------
    # C. Supervisor调度
    # ----------------------------------------

    def router(state: AgentState):

        return {
            "phase": "dispatching"
        }

    def route_task(
        state: AgentState,
    ):

        tasks = state["plan"]["tasks"]

        index = state.get(
            "cursor",
            0,
        )

        if index >= len(tasks):
            return "coaching"

        return tasks[index]["worker"]

    # ----------------------------------------
    # D. 为当前任务准备执行次数
    # ----------------------------------------

    def prepare_task(
        state: AgentState,
    ):

        task = TaskSpec.model_validate(
            state["plan"]["tasks"][
                state["cursor"]
            ]
        )

        attempts = dict(
            state.get("attempts", {})
        )

        attempt = (
            attempts.get(task.task_id, 0)
            + 1
        )

        if attempt > 2:

            raise RuntimeError(
                "超出Worker最大执行次数"
            )

        attempts[
            task.task_id
        ] = attempt

        return task, attempt, attempts

    # ----------------------------------------
    # E. Knowledge节点
    # ----------------------------------------

    def knowledge_node(
        state: AgentState,
    ):

        task, attempt, attempts = (
            prepare_task(state)
        )

        msg = knowledge_agent.execute(
            task,
            attempt,
        )

        return {
            "pending": msg.model_dump(),

            "attempts": attempts,

            "phase": "knowledge_done",
        }

    # ----------------------------------------
    # F. Evidence验证与交接
    # ----------------------------------------

    def evidence_node(
        state: AgentState,
    ):

        original = Handoff.model_validate(
            state["pending"]
        )

        verified = evidence_agent.execute(
            original
        )

        return {
            "handoffs": (
                state.get(
                    "handoffs",
                    [],
                )
                + [original.model_dump()]
            ),

            "pending": (
                verified.model_dump()
            ),

            "phase": "evidence_checked",
        }

    # ----------------------------------------
    # G. Practice节点
    # ----------------------------------------

    def practice_node(
        state: AgentState,
    ):

        task, attempt, attempts = (
            prepare_task(state)
        )

        msg = practice_agent.execute(
            task,
            state["goal"],
            attempt,
        )

        return {
            "pending": msg.model_dump(),

            "attempts": attempts,

            "phase": "practice_done",
        }

    # ----------------------------------------
    # H. 统一任务交接中心
    # ----------------------------------------

    def handoff_node(
        state: AgentState,
    ):

        msg = Handoff.model_validate(
            state["pending"]
        )

        task = TaskSpec.model_validate(
            state["plan"]["tasks"][
                state["cursor"]
            ]
        )

        # 不能接收其他任务伪装的结果
        if (
            msg.task_id != task.task_id
            or msg.recipient != "supervisor"
        ):

            raise ValueError(
                "交接任务ID或接收方"
                "与当前任务不匹配"
            )

        messages = (
            state.get(
                "handoffs",
                [],
            )
            + [msg.model_dump()]
        )

        attempts = state["attempts"].get(
            task.task_id,
            0,
        )

        # 只有error才自动重试，
        # 空结果、不可用不是同一类故障。
        retry = (
            msg.status == "error"
            and attempts < 2
        )

        if retry:

            # 不推进cursor，
            # 下次router重新派发当前任务。
            return {
                "handoffs": messages,

                "retry_current": True,

                "phase": "retrying",
            }

        result = {
            "task_id": msg.task_id,

            "worker": task.worker,

            "status": msg.status,

            **msg.payload,
        }

        return {
            "handoffs": messages,

            "retry_current": False,

            "worker_results": (
                state.get(
                    "worker_results",
                    [],
                )
                + [result]
            ),

            "cursor": (
                state["cursor"] + 1
            ),

            "phase": "task_delivered",
        }

    # ----------------------------------------
    # I. Coaching综合节点
    # ----------------------------------------

    def coaching_node(
        state: AgentState,
    ):

        reply = coaching_agent.execute(
            state["question"],
            state["goal"],

            state.get(
                "worker_results",
                [],
            ),
        )

        return {
            "answer": reply.answer,

            "evidence_ids": (
                reply.evidence_ids
            ),

            "limitations": (
                reply.limitations
            ),

            "phase": "completed",
        }

    # ----------------------------------------
    # J. 用户拒绝计划
    # ----------------------------------------

    def cancel_node(
        state: AgentState,
    ):

        return {
            "phase": "cancelled",

            "answer": (
                "用户拒绝本轮计划，"
                "未执行专门Agent任务。"
            ),
        }

    # ========================================
    # 8. 将Agent接入LangGraph
    # ========================================

    builder = StateGraph(
        AgentState
    )

    for name, fn in (
        ("supervisor", supervisor),
        ("approval", approval),
        ("router", router),
        ("knowledge", knowledge_node),
        ("evidence", evidence_node),
        ("practice", practice_node),
        ("handoff", handoff_node),
        ("coaching", coaching_node),
        ("cancel", cancel_node),
    ):

        builder.add_node(
            name,
            fn,
        )

    # ========================================
    # 9. 注册路线
    # ========================================

    builder.add_edge(
        START,
        "supervisor",
    )

    builder.add_edge(
        "supervisor",
        "approval",
    )

    builder.add_conditional_edges(
        "approval",
        after_approval,

        {
            "router": "router",
            "cancel": "cancel",
        },
    )

    builder.add_conditional_edges(
        "router",
        route_task,

        {
            "knowledge": "knowledge",
            "practice": "practice",
            "coaching": "coaching",
        },
    )

    # Knowledge先交给Evidence
    builder.add_edge(
        "knowledge",
        "evidence",
    )

    # Evidence再交给Supervisor
    builder.add_edge(
        "evidence",
        "handoff",
    )

    # Practice直接交给Supervisor
    builder.add_edge(
        "practice",
        "handoff",
    )

    # Supervisor收到结果后再次调度
    builder.add_edge(
        "handoff",
        "router",
    )

    builder.add_edge(
        "coaching",
        END,
    )

    builder.add_edge(
        "cancel",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer
    )
