from __future__ import annotations

import hashlib

from .capture_stage import CaptureStage
from .knowledge_stage import KnowledgeStage
from .models import (
    BoundaryKind,
    ConversationIntent,
    DecisionRecord,
    DiscoveryBoundary,
    InterviewState,
    KnowledgeStatus,
    TurnCapture,
)
from .question_stage import QuestionStage
from .requirement_stage import (
    ContradictionStage,
    ImplicationStage,
    RequirementStage,
)
from .runtime import record_message
from .store import save_state


class AdaptivePMEngine:
    """Orchestrate independent adaptive-discovery reasoning stages.

    Python owns durable state, provenance, IDs and protocol invariants.
    Each semantic decision has its own model contract.
    """

    def __init__(self):
        self.capture_stage = CaptureStage()
        self.knowledge_stage = KnowledgeStage()
        self.requirement_stage = RequirementStage()
        self.implication_stage = ImplicationStage()
        self.contradiction_stage = ContradictionStage()
        self.question_stage = QuestionStage()

    @staticmethod
    def new_state(raw_idea: str) -> InterviewState:
        return InterviewState(raw_idea=raw_idea)

    @staticmethod
    def _apply_recommendation_decisions(
        state: InterviewState,
        capture: TurnCapture,
    ) -> None:
        for identity in capture.accepted_recommendation_ids:
            existing = state.recommendations.get(identity)
            if existing:
                state.recommendations[identity] = (
                    existing.model_copy(update={
                        "status": KnowledgeStatus.CONFIRMED
                    })
                )

        for identity in capture.rejected_recommendation_ids:
            existing = state.recommendations.get(identity)
            if existing:
                state.recommendations[identity] = (
                    existing.model_copy(update={
                        "status": KnowledgeStatus.REJECTED
                    })
                )

    @staticmethod
    def _mark_previous_decision_from_control(
        state: InterviewState,
        intent: ConversationIntent,
    ) -> None:
        key = state.last_decision_key
        if not key or key not in state.decisions:
            return

        if intent in (
            ConversationIntent.OBJECTION,
            ConversationIntent.DESIGN_DEFERRAL,
        ):
            resolution = "REJECTED"
        elif intent == ConversationIntent.UNCERTAINTY:
            resolution = "DEFERRED"
        else:
            return

        state.decisions[key] = state.decisions[key].model_copy(
            update={"resolution": resolution}
        )

    @staticmethod
    def _fallback_boundary(
        state: InterviewState,
        capture: TurnCapture,
        user_message: str,
    ) -> DiscoveryBoundary | None:
        """Persist model-classified interview feedback even if boundary fields were omitted."""
        if capture.intent == ConversationIntent.OBJECTION:
            kind = BoundaryKind.REJECTED_INQUIRY
            instruction = (
                "Do not retry, paraphrase, or deepen the immediately "
                "rejected inquiry. Choose a materially different "
                "product decision."
            )
        elif capture.intent == ConversationIntent.DESIGN_DEFERRAL:
            kind = BoundaryKind.DESIGN_DEFERRAL
            instruction = (
                "Do not ask the founder for UI, navigation, layout, "
                "or designer-level implementation detail for this "
                "decision unless a product rule genuinely depends on it."
            )
        elif (
            capture.intent == ConversationIntent.UNCERTAINTY
            and state.last_decision_key
        ):
            kind = BoundaryKind.DEFERRED_DECISION
            instruction = (
                "This decision was explicitly left undecided. Treat it "
                "as deferred and do not pressure the founder to resolve "
                "it unless later product knowledge creates a genuine "
                "blocking dependency."
            )
        else:
            return None

        digest = hashlib.sha1(
            (
                f"{state.turn_count}|{kind.value}|"
                f"{state.last_decision_key}|{user_message}"
            ).encode("utf-8")
        ).hexdigest()[:16]

        return DiscoveryBoundary(
            id=f"boundary_{digest}",
            kind=kind,
            subject=state.last_decision_key or "current inquiry",
            instruction=instruction,
            evidence=user_message,
            source_turn=state.turn_count,
            decision_key=state.last_decision_key,
        )

    def _persist_boundary(
        self,
        state: InterviewState,
        capture: TurnCapture,
        user_message: str,
    ) -> None:
        boundary = self.capture_stage.boundary_from_capture(
            state,
            capture,
        )
        if boundary is None:
            boundary = self._fallback_boundary(
                state,
                capture,
                user_message,
            )
        if (
            boundary is not None
            and all(
                existing.id != boundary.id
                for existing in state.boundaries
            )
        ):
            state.boundaries.append(boundary)

    @staticmethod
    def _blocking_contradiction(state: InterviewState) -> bool:
        return any(
            item.blocking and not item.resolved
            for item in state.contradictions.values()
        )

    def process_turn(
        self,
        state: InterviewState,
        user_message: str,
    ) -> str:
        state.turn_count += 1
        record_message(state, "founder", user_message)

        # 1. Capture explicit founder meaning and conversation intent.
        capture = self.capture_stage.capture(
            state,
            user_message,
        )
        self._persist_boundary(
            state,
            capture,
            user_message,
        )
        self._apply_recommendation_decisions(
            state,
            capture,
        )

        # Clarification/rationale are conversation control, not product facts.
        if capture.intent in (
            ConversationIntent.CLARIFICATION,
            ConversationIntent.RATIONALE_REQUEST,
        ):
            response = self.capture_stage.control_response(
                state,
                capture.intent,
                user_message,
            )
            record_message(state, "pm", response)
            save_state(state)
            return response

        self._mark_previous_decision_from_control(
            state,
            capture.intent,
        )

        if (
            capture.confirms_previous_answer
            and state.last_decision_key in state.decisions
        ):
            prior = state.decisions[state.last_decision_key]
            state.decisions[state.last_decision_key] = (
                prior.model_copy(update={
                    "resolution": "ANSWERED",
                    "answer_summary": user_message.strip(),
                })
            )

        # 2. Semantic grounding: does the current founder evidence
        # actually support each captured proposition?
        observations = self.capture_stage.semantic_ground(
            state,
            capture,
            user_message,
        )
        existing_observation_ids = {
            item.id for item in state.observations
        }
        state.observations.extend(
            item
            for item in observations
            if item.id not in existing_observation_ids
        )

        # 3. Canonicalize/classify grounded knowledge.
        canonical = self.knowledge_stage.canonicalize(
            state,
            observations,
            capture.intent,
        )
        changed_keys = self.knowledge_stage.apply(
            state,
            canonical,
        )

        # 4. Independently update dynamic requirements,
        # coverage/depth and dependency graph.
        requirement_result = self.requirement_stage.reason(
            state,
            changed_keys,
        )
        self.requirement_stage.apply(
            state,
            requirement_result,
        )

        # 5. Derive proposed implications separately from facts.
        implication_result = self.implication_stage.derive(
            state,
            changed_keys,
        )
        self.implication_stage.apply(
            state,
            implication_result,
        )

        # 6. Check current canonical facts for contradictions.
        contradiction_result = self.contradiction_stage.detect(
            state,
            changed_keys,
        )
        self.contradiction_stage.apply(
            state,
            contradiction_result,
        )

        # 7. Generate candidate decisions without ranking them.
        candidate_result = self.question_stage.generate(
            state,
            capture.intent,
        )
        self.question_stage.persist_advice(
            state,
            candidate_result,
        )

        if (
            candidate_result.completion.complete
            and not self._blocking_contradiction(state)
        ):
            state.complete = True
            state.last_question = None
            state.last_decision_key = None
            response = (
                "Discovery is sufficiently complete for the current "
                "product scope."
            )
            record_message(state, "pm", response)
            save_state(state)
            return response

        # 8. Suppress persistent semantic repeats.
        eligible = self.question_stage.eligible_candidates(
            candidate_result,
            state,
        )

        # 9. Prioritize independently by information value.
        prioritized = self.question_stage.prioritize(
            state,
            eligible,
        )

        # 10. Final semantic audit; at most one safe rewrite.
        selected = self.question_stage.select(
            state,
            prioritized,
        )

        if selected is None:
            state.last_question = None
            state.last_decision_key = None
            response = (
                "I have enough on the current line of discovery for "
                "now. I’ll leave the remaining low-value details open."
            )
            record_message(state, "pm", response)
            save_state(state)
            return response

        candidate, question = selected
        state.last_question = question
        state.last_decision_key = candidate.decision_key
        state.decisions[candidate.decision_key] = DecisionRecord(
            decision_key=candidate.decision_key,
            requirement_ids=candidate.requirement_ids,
            question=question,
            asked_turn=state.turn_count,
            resolution="OPEN",
        )

        if candidate_result.advice_options:
            advice = "\n".join(
                f"- {item.statement}"
                for item in candidate_result.advice_options[:3]
            )
            response = f"{advice}\n\n{question}"
        else:
            response = question

        record_message(state, "pm", response)
        save_state(state)
        return response

    def start(
        self,
        raw_idea: str,
    ) -> tuple[InterviewState, str]:
        state = self.new_state(raw_idea)
        save_state(state)
        response = self.process_turn(state, raw_idea)
        return state, response
