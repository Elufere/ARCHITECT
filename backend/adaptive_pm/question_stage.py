from __future__ import annotations

from agents.llm import get_structured_model
from .models import (
    CandidateGenerationResult,
    ConversationIntent,
    InterviewState,
    KnowledgeStatus,
    PriorityResult,
    QuestionAudit,
    QuestionCandidate,
    RecommendationRecord,
)
from .prompts import QUESTION_AUDIT_PROMPT
from .runtime import invoke_structured
from .stage_prompts import CANDIDATE_PROMPT, PRIORITY_PROMPT


class QuestionStage:
    """Generate, prioritize and audit questions as separate semantic decisions."""

    def __init__(self):
        self.candidate_model = get_structured_model(
            call_name="adaptive_pm.candidates",
            schema=CandidateGenerationResult,
            max_tokens=3000,
        )
        self.priority_model = get_structured_model(
            call_name="adaptive_pm.priority",
            schema=PriorityResult,
            max_tokens=800,
        )
        self.audit_model = get_structured_model(
            call_name="adaptive_pm.question_audit",
            schema=QuestionAudit,
            max_tokens=900,
        )

    def generate(
        self,
        state: InterviewState,
        intent: ConversationIntent,
    ) -> CandidateGenerationResult:
        payload = {
            "turn": state.turn_count,
            "founder_requested_advice": (
                intent == ConversationIntent.ADVICE_REQUEST
            ),
            "state": state.compact_context(),
        }
        return invoke_structured(
            self.candidate_model,
            CANDIDATE_PROMPT,
            payload,
            CandidateGenerationResult,
        )

    @staticmethod
    def persist_advice(
        state: InterviewState,
        result: CandidateGenerationResult,
    ) -> None:
        for option in result.advice_options:
            state.recommendations[option.id] = RecommendationRecord(
                id=option.id,
                statement=option.statement,
                requirement_ids=option.requirement_ids,
                status=KnowledgeStatus.PROPOSED,
                source_turn=state.turn_count,
            )

    @staticmethod
    def eligible_candidates(
        result: CandidateGenerationResult,
        state: InterviewState,
    ) -> list[QuestionCandidate]:
        seen_decisions: set[str] = set()
        eligible: list[QuestionCandidate] = []

        for candidate in result.candidates:
            if (
                not candidate.eligible
                or candidate.decision_key in seen_decisions
            ):
                continue

            seen_decisions.add(candidate.decision_key)
            prior = state.decisions.get(candidate.decision_key)
            if (
                prior
                and prior.resolution
                in ("ANSWERED", "DEFERRED", "REJECTED")
            ):
                continue

            eligible.append(candidate)

        return eligible

    def prioritize(
        self,
        state: InterviewState,
        candidates: list[QuestionCandidate],
    ) -> list[QuestionCandidate]:
        if len(candidates) <= 1:
            return candidates

        payload = {
            "turn": state.turn_count,
            "candidates": [
                item.model_dump(mode="json")
                for item in candidates
            ],
            "requirements": [
                {
                    "id": item.id,
                    "label": item.label,
                    "status": item.status.value,
                    "coverage": item.coverage,
                    "depth": item.depth,
                    "high_impact": item.high_impact,
                }
                for item in state.requirements.values()
            ],
            "recent_conversation": state.recent_messages[-4:],
        }
        result = invoke_structured(
            self.priority_model,
            PRIORITY_PROMPT,
            payload,
            PriorityResult,
        )

        by_id = {item.id: item for item in candidates}
        ordered: list[QuestionCandidate] = []

        for identity in result.ordered_candidate_ids:
            candidate = by_id.get(identity)
            if candidate is not None and candidate not in ordered:
                ordered.append(candidate)

        # Protocol omissions must not silently delete an eligible candidate.
        ordered.extend(
            item for item in candidates if item not in ordered
        )
        return ordered

    def audit(
        self,
        state: InterviewState,
        candidate: QuestionCandidate,
    ) -> QuestionAudit:
        payload = {
            "candidate": candidate.model_dump(mode="json"),
            "question": candidate.question,
            "knowledge": [
                {
                    "key": item.key,
                    "statement": item.statement,
                    "status": item.status.value,
                }
                for item in state.knowledge.values()
            ],
            "decision_history": [
                item.model_dump(mode="json")
                for item in state.decisions.values()
            ],
            "boundaries": [
                item.model_dump(mode="json")
                for item in state.boundaries
            ],
            "recent_conversation": state.recent_messages[-6:],
        }
        return invoke_structured(
            self.audit_model,
            QUESTION_AUDIT_PROMPT,
            payload,
            QuestionAudit,
        )

    @staticmethod
    def _valid_question_shape(question: str) -> bool:
        text = question.strip()
        return bool(
            text
            and text.endswith("?")
            and text.count("?") == 1
        )

    def select(
        self,
        state: InterviewState,
        candidates: list[QuestionCandidate],
    ) -> tuple[QuestionCandidate, str] | None:
        for candidate in candidates[:3]:
            audit = self.audit(state, candidate)

            if audit.reject_candidate:
                continue

            if (
                audit.passed
                and self._valid_question_shape(candidate.question)
            ):
                return candidate, candidate.question.strip()

            # A single rewrite is allowed only for wording/granularity.
            # The rewrite must pass the same semantic audit before delivery.
            if audit.revised_question:
                revised = candidate.model_copy(update={
                    "question": audit.revised_question.strip()
                })
                second = self.audit(state, revised)
                if (
                    second.passed
                    and not second.reject_candidate
                    and self._valid_question_shape(revised.question)
                ):
                    return candidate, revised.question

        return None
