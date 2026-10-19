
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from app.agent.schema import RouteDecision

from app.assessment.schema import AssessmentDraft

from app.assessment.cross_schema import (
    CrossAssessmentDraft,
    CrossCandidate,
)

from app.assessment.verification import (
    VerificationJudgment,
)

from app.memory.schema import (
    AgentSession,
    ExperienceRecord,
    VerificationSession,
    VerificationTurn,
)

from app.memory.store import experience_fingerprint

from app.rag.pipeline import (
    RAGCitation,
    RAGResponse,
)


# ============================================
# 1. Agent回复
# ============================================

@dataclass(frozen=True)
class AgentReply:
    message: str
    mode: str

    citations: list[RAGCitation] = field(
        default_factory=list
    )

    status: str | None = None


# ============================================
# 2. 外部模块接口
# ============================================

class Router(Protocol):
    def route(
        self,
        message: str,
        stage: str,
    ) -> RouteDecision:
        ...


class KnowledgeTool(Protocol):
    def answer(
        self,
        question: str,
    ) -> RAGResponse:
        ...


class Assessor(Protocol):
    def assess(
        self,
        experience: ExperienceRecord,
    ) -> AssessmentDraft:
        ...


class CrossAssessor(Protocol):
    def assess(
        self,
        experiences: Sequence[ExperienceRecord],
        assessments: Sequence[AssessmentDraft | None] | None = None,
    ) -> CrossAssessmentDraft:
        ...


class Verifier(Protocol):
    def assess(
        self,
        candidate: CrossCandidate,
        question: str,
        answer: str,
        previous_turns: Sequence[VerificationTurn] = (),
    ) -> VerificationJudgment:
        ...


# ============================================
# 3. 核心Agent
# ============================================

class AgentOrchestrator:

    ACTIVE_STAGES = {
        "awaiting_situation",
        "awaiting_action",
        "awaiting_outcome",
    }

    MAX_VERIFY_ROUNDS = 3

    def __init__(
        self,
        router: Router,
        knowledge_tool: KnowledgeTool,
        assessor: Assessor | None = None,
        cross_assessor: CrossAssessor | None = None,
        verifier: Verifier | None = None,
    ):
        self.router = router
        self.knowledge_tool = knowledge_tool
        self.assessor = assessor
        self.cross_assessor = cross_assessor
        self.verifier = verifier

    # ========================================
    # 4. 用户消息统一入口
    # ========================================

    def handle(
        self,
        message: str,
        session: AgentSession,
    ) -> AgentReply:

        message = message.strip()

        if not message:
            raise ValueError("用户消息不能为空")

        # A. 教材知识问答
        if message == "/ask":
            return AgentReply(
                "请在 /ask 后面输入教材问题。",
                "question",
            )

        if message.startswith("/ask "):
            question = message[5:].strip()

            if not question:
                raise ValueError(
                    "教材问题不能为空"
                )

            return self._answer_knowledge(question)

        # B. 开始新经历访谈
        if message == "/start":
            return self._start_exploration(session)

        # C. 跨经历分析
        if message == "/compare":
            return self.compare_experiences(session)

        # D. 开始验证某个候选方向
        if message == "/verify":
            return AgentReply(
                "请输入 /verify 1 来选择跨经历报告"
                "中的第1个候选方向。",
                "question",
            )

        if message.startswith("/verify "):
            parts = message.split()

            if len(parts) != 2:
                return AgentReply(
                    "用法：/verify 1",
                    "question",
                )

            try:
                number = int(parts[1])
            except ValueError:
                return AgentReply(
                    "候选编号必须是整数，例如 /verify 1。",
                    "question",
                )

            return self.start_verification(
                number,
                session,
            )

        # E. 重试已经保存的验证回答
        if message == "/retryverify":
            return self.retry_verification(session)

        # F. 结束验证
        if message == "/cancelverify":

            if session.stage != "awaiting_verification":
                return AgentReply(
                    "当前没有正在进行的假设验证。",
                    "question",
                )

            session.stage = "idle"
            session.verification = None

            return AgentReply(
                "已结束当前验证流程，"
                "原始经历与跨经历报告仍保留。",
                "verification_cancelled",
            )

        # G. 未知命令不能写成用户经历
        if message.startswith("/"):
            return AgentReply(
                "未知命令。可使用 /start、"
                "/ask、/compare、/verify 1。",
                "unsupported",
            )

        # H. 验证阶段优先处理用户回答
        if session.stage == "awaiting_verification":
            return self._handle_verification_answer(
                message,
                session,
            )

        # I. 三阶段经历访谈
        if session.stage in self.ACTIVE_STAGES:
            return self._handle_experience(
                message,
                session,
            )

        # J. 空闲时才调用自然语言Router
        decision = self.router.route(
            message=message,
            stage=session.stage,
        )

        if decision.action == "knowledge":
            return self._answer_knowledge(message)

        if decision.action == "start_exploration":
            return self._start_exploration(session)

        if decision.action == "experience_reply":
            return AgentReply(
                "当前没有正在进行的访谈，"
                "请先输入 /start。",
                "question",
            )

        return AgentReply(
            "当前支持教材问答、经历收集和优势验证。"
            "该请求暂无对应工具。",
            "unsupported",
        )

    # ========================================
    # 5. 开始候选假设验证
    # ========================================

    def start_verification(
        self,
        number: int,
        session: AgentSession,
    ) -> AgentReply:

        if session.stage != "idle":
            return AgentReply(
                "当前已有进行中的访谈或验证。"
                "请先完成，或使用 /reset 放弃草稿、"
                "/cancelverify 结束验证。",
                "question",
            )

        report = session.cross_report

        if report is None:
            return AgentReply(
                "没有已保存的跨经历报告。"
                "请先完成至少两条经历并输入 /compare。",
                "verification_unavailable",
            )

        current = experience_fingerprint(
            session.experiences
        )

        if session.cross_report_fingerprint != current:

            session.cross_report = None
            session.cross_report_fingerprint = None

            return AgentReply(
                "历史经历已经变化，"
                "请重新执行 /compare。",
                "verification_unavailable",
            )

        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or not 1 <= number <= len(report.candidates)
        ):
            return AgentReply(
                f"候选方向编号无效。当前报告有"
                f"{len(report.candidates)}个候选方向。",
                "verification_unavailable",
            )

        if self.verifier is None:
            return AgentReply(
                "尚未接入验证分析器。",
                "verification_unavailable",
            )

        candidate = report.candidates[number - 1]

        # 首轮问题由程序构造
        question = (
            f"我们正在验证候选方向“{candidate.name}”。"
            "请回想一个不同于之前已记录经历的具体情境："
            "当时你是否采取过类似行动？"
            "你本人做了什么，最后结果如何？"
            "如果没有、没有成功或者主要由他人完成，"
            "也可以如实说明。"
        )

        session.verification = VerificationSession(
            candidate_number=number,
            candidate_name=candidate.name,
            source_fingerprint=current,
            pending_question=question,
        )

        session.stage = "awaiting_verification"

        return AgentReply(
            question
            + "\n\n你可以直接回答；"
            "输入 /cancelverify 可以退出。",
            "verification_question",
        )

    # ========================================
    # 6. 接收验证回答
    # ========================================

    def _handle_verification_answer(
        self,
        answer: str,
        session: AgentSession,
    ) -> AgentReply:

        state = session.verification

        if state is None:
            raise RuntimeError(
                "验证状态缺失"
            )

        # 如果上次模型失败，不允许覆盖旧回答
        if state.pending_answer is not None:
            return AgentReply(
                "上一条回答已保存但分析失败。"
                "请使用 /retryverify 重试，"
                "或 /cancelverify 结束。",
                "verification_waiting_retry",
            )

        if len(answer) < 8:
            return AgentReply(
                "请描述一个具体事实，"
                "可以包含你的行动、他人参与情况和结果。",
                "verification_question",
            )

        # 先暂存回答，再调用模型
        state.pending_answer = answer

        return self._review_pending_answer(
            session
        )

    # ========================================
    # 7. 重试未完成的验证分析
    # ========================================

    def retry_verification(
        self,
        session: AgentSession,
    ) -> AgentReply:

        state = session.verification

        if (
            session.stage != "awaiting_verification"
            or state is None
            or state.pending_answer is None
        ):
            return AgentReply(
                "当前没有等待重试的验证回答。",
                "question",
            )

        return self._review_pending_answer(
            session
        )

    # ========================================
    # 8. 分析暂存回答并更新验证状态
    # ========================================

    def _review_pending_answer(
        self,
        session: AgentSession,
    ) -> AgentReply:

        state = session.verification
        report = session.cross_report

        if (
            state is None
            or report is None
            or state.pending_answer is None
        ):
            raise RuntimeError(
                "缺少验证数据"
            )

        if (
            experience_fingerprint(session.experiences)
            != state.source_fingerprint
        ):
            raise ValueError(
                "原始经历已改变，需要重新 /compare"
            )

        if not 1 <= state.candidate_number <= len(
            report.candidates
        ):
            raise ValueError(
                "候选方向编号已失效"
            )

        if self.verifier is None:
            raise RuntimeError(
                "尚未接入验证分析器"
            )

        candidate = report.candidates[
            state.candidate_number - 1
        ]

        try:
            result = self.verifier.assess(
                candidate=candidate,
                question=state.pending_question,
                answer=state.pending_answer,
                previous_turns=state.turns,
            )

        except Exception as exc:
            return AgentReply(
                "本轮回答已暂存，但证据分析失败："
                f"{type(exc).__name__}: {exc}\n"
                "不需要重新输入回答，"
                "请输入 /retryverify。",
                "verification_error",
            )

        # 分析成功后才正式保存这一轮判断
        state.turns.append(
            VerificationTurn(
                question=state.pending_question,
                answer=state.pending_answer,
                relation=result.relation,
                quote=result.quote,
                reason=result.reason,
            )
        )

        state.pending_answer = None

        descriptions = {
            "supports": "新增支持线索",
            "challenges": "可能的限制或反例线索",
            "unclear": "目前信息不足",
        }

        message = (
            "【本轮证据判断】"
            f"{descriptions[result.relation]}\n"
            "原文："
            f"{result.quote or '没有可引用的明确片段'}\n"
            f"解释：{result.reason}\n"
            "注意：这是对用户自述的初步分析，"
            "不代表已验证的事实。"
        )

        # 最多三轮
        if len(state.turns) >= self.MAX_VERIFY_ROUNDS:

            state.completed = True
            state.pending_question = ""

            session.stage = "idle"

            return AgentReply(
                message
                + "\n\n已完成3轮验证。"
                "输入 /verification 查看证据记录。",
                "verification_complete",
            )

        # 动态使用模型提出的下一轮问题
        state.pending_question = result.next_question

        return AgentReply(
            message
            + "\n\n【下一轮问题】\n"
            + state.pending_question,
            "verification_question",
        )

    # ========================================
    # 9. 跨经历分析
    # ========================================

    def compare_experiences(
        self,
        session: AgentSession,
    ) -> AgentReply:

        if session.stage == "awaiting_verification":
            return AgentReply(
                "当前正在验证候选方向，"
                "请先完成或 /cancelverify。",
                "question",
            )

        count = len(session.experiences)

        if count < 2:
            return AgentReply(
                f"目前只有{count}条完整经历。"
                "至少需要2条，输入 /start 继续收集。",
                "compare_insufficient",
            )

        if self.cross_assessor is None:
            return AgentReply(
                "尚未接入跨经历分析模块。",
                "compare_error",
            )

        try:
            report = self.cross_assessor.assess(
                session.experiences,
                session.assessments,
            )

        except Exception as exc:
            return AgentReply(
                "跨经历分析失败："
                f"{type(exc).__name__}: {exc}。"
                "原始经历未修改。",
                "compare_error",
            )

        # 保存跨经历报告，供后面的/verify使用
        session.cross_report = report

        session.cross_report_fingerprint = (
            experience_fingerprint(
                session.experiences
            )
        )

        # 新报告使旧的验证状态失效
        session.verification = None

        return self._format_cross_report(
            report,
            count,
        )

    # ========================================
    # 10. 重新分析单条经历
    # ========================================

    def reanalyze_experience(
        self,
        number: int,
        session: AgentSession,
    ) -> AgentReply:

        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or not 1 <= number <= len(session.experiences)
        ):
            raise ValueError(
                "经历编号不存在"
            )

        if session.stage == "awaiting_verification":
            return AgentReply(
                "请先完成或取消当前验证，"
                "再重新分析单条经历。",
                "question",
            )

        if self.assessor is None:
            return AgentReply(
                "未接入Assessment模块。"
                "仅凭一次经历不能确认稳定优势。",
                "assessment_error",
            )

        try:
            assessment = self.assessor.assess(
                session.experiences[number - 1]
            )

        except Exception as exc:
            return AgentReply(
                f"第{number}条经历重新分析失败："
                f"{type(exc).__name__}: {exc}。"
                "原始经历未修改。",
                "assessment_error",
            )

        session.assessments[number - 1] = assessment

        # 单经历分析变化，旧综合报告需要重算
        session.cross_report = None
        session.cross_report_fingerprint = None
        session.verification = None

        reply = self._format_assessment(
            assessment
        )

        return AgentReply(
            f"第{number}条经历重新分析成功。\n\n"
            + reply.message,
            "assessment",
        )

    # ========================================
    # 11. RAG工具调用
    # ========================================

    def _answer_knowledge(
        self,
        question: str,
    ) -> AgentReply:

        result = self.knowledge_tool.answer(
            question
        )

        return AgentReply(
            result.answer,
            "knowledge",
            citations=result.citations,
            status=result.status,
        )

    # ========================================
    # 12. 启动新的经历访谈
    # ========================================

    def _start_exploration(
        self,
        session: AgentSession,
    ) -> AgentReply:

        if session.stage != "idle":
            return AgentReply(
                "已有正在进行的访谈或验证。"
                "请先完成，或使用 /reset 放弃当前草稿。",
                "question",
            )

        session.stage = "awaiting_situation"
        session.draft.clear()

        return AgentReply(
            "请回想一次真实经历。"
            "可以是学习、研究、工作或生活中的事。\n"
            "当时发生了什么？",
            "question",
        )

    # ========================================
    # 13. 三阶段经历收集
    # ========================================

    def _handle_experience(
        self,
        message: str,
        session: AgentSession,
    ) -> AgentReply:

        if len(message) < 8:
            return AgentReply(
                "请补充更具体的事实、行动或结果。",
                "question",
            )

        if session.stage == "awaiting_situation":

            session.draft["situation"] = message
            session.stage = "awaiting_action"

            return AgentReply(
                "已记录情境。"
                "你本人具体采取了哪些行动？",
                "question",
            )

        if session.stage == "awaiting_action":

            session.draft["action"] = message
            session.stage = "awaiting_outcome"

            return AgentReply(
                "已记录行动。"
                "最终结果如何？"
                "有哪些能够观察的变化？",
                "question",
            )

        if session.stage == "awaiting_outcome":

            situation = session.draft.get(
                "situation"
            )

            action = session.draft.get(
                "action"
            )

            if not situation or not action:
                raise RuntimeError(
                    "访谈草稿缺少情境或行动"
                )

            record = ExperienceRecord(
                situation=situation,
                action=action,
                outcome=message,
            )

            # 保存原始经历
            session.experiences.append(record)
            session.assessments.append(None)

            session.stage = "idle"
            session.draft.clear()

            # 新经历使旧报告和验证失效
            session.cross_report = None
            session.cross_report_fingerprint = None
            session.verification = None

            if self.assessor is None:
                return AgentReply(
                    "已记录完整经历。"
                    "仅凭一次经历不能确定稳定的个人优势。",
                    "recorded",
                )

            try:
                assessment = self.assessor.assess(
                    record
                )

            except Exception as exc:
                return AgentReply(
                    "经历已记录，但Assessment失败："
                    f"{type(exc).__name__}: {exc}。"
                    "可使用 /reanalyze 编号重试。",
                    "assessment_error",
                )

            session.assessments[-1] = assessment

            return self._format_assessment(
                assessment
            )

        raise RuntimeError(
            f"未知访谈阶段：{session.stage}"
        )

    # ========================================
    # 14. 展示单经历结果
    # ========================================

    def _format_assessment(
        self,
        assessment: AssessmentDraft,
    ) -> AgentReply:

        parts = [
            "以下是用户自述的初步分析。"
            "仅凭一次经历不能确定稳定优势。"
        ]

        if assessment.observations:

            parts.append("\n【行为观察】")

            for i, obs in enumerate(
                assessment.observations,
                1,
            ):
                parts.append(
                    f"{i}. {obs.description}\n"
                    f"   用户原话：{obs.quote}"
                )

        else:
            parts.append(
                "\n【行为观察】现有信息不足"
            )

        if assessment.hypotheses:

            parts.append(
                "\n【待验证的优势方向】"
            )

            for i, hyp in enumerate(
                assessment.hypotheses,
                1,
            ):
                parts.append(
                    f"{i}. {hyp.name}："
                    f"{hyp.interpretation}"
                )

        else:
            parts.append(
                "\n【待验证的优势方向】现有信息不足"
            )

        parts.append(
            "\n【下一轮建议问题】\n"
            + assessment.follow_up_question
        )

        parts.append(
            "\n【分析限制】\n"
            + assessment.limitation
        )

        return AgentReply(
            "\n".join(parts),
            "assessment",
        )

    # ========================================
    # 15. 展示跨经历报告
    # ========================================

    def _format_cross_report(
        self,
        report: CrossAssessmentDraft,
        count: int,
    ) -> AgentReply:

        parts = [
            f"已综合分析{count}条历史经历。"
            "这些只是候选方向，并非稳定优势结论。"
        ]

        if not report.candidates:
            parts.append(
                "\n【重复行为模式】"
                "目前没有足够的跨经历证据形成候选方向。"
            )

        for i, candidate in enumerate(
            report.candidates,
            1,
        ):

            parts.extend([
                f"\n【候选方向 {i}】{candidate.name}",
                candidate.interpretation,
                "支持证据：",
            ])

            for item in candidate.supports:
                parts.append(
                    f"  经历{item.experience_number}"
                    f"（{item.source_field}）："
                    f"{item.quote}"
                )

            if candidate.possible_counterevidence:

                parts.append(
                    "可能的限制或反例线索（未确认）："
                )

                for item in candidate.possible_counterevidence:
                    parts.append(
                        f"  经历{item.experience_number}"
                        f"（{item.source_field}）："
                        f"{item.quote}"
                    )

            parts.append(
                "不确定性：" + candidate.uncertainty
            )

        parts.append(
            "\n【下一轮建议问题】\n"
            + report.next_question
        )

        parts.append(
            "\n【总体限制】\n"
            + report.limitations
        )

        if report.candidates:
            parts.append(
                "\n输入 /verify 1 开始验证第1个方向"
                f"（当前有{len(report.candidates)}个候选方向）。"
            )

        return AgentReply(
            "\n".join(parts),
            "cross_assessment",
        )
