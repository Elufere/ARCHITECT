from __future__ import annotations

import hashlib
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm import get_chat_model, get_structured_model
from .models import (
    ConversationIntent,
    DecisionRecord,
    DiscoveryBoundary,
    FactSource,
    GroundingAudit,
    InterviewState,
    KnowledgeRecord,
    KnowledgeStatus,
    Observation,
    PlanningResult,
    ProductConcept,
    QuestionAudit,
    QuestionCandidate,
    ReasoningUpdate,
    RequirementRecord,
    TurnCapture,
)
from .prompts import (
    CAPTURE_PROMPT,
    CONTROL_RESPONSE_PROMPT,
    GROUNDING_PROMPT,
    PLANNING_PROMPT,
    QUESTION_AUDIT_PROMPT,
    REASONING_PROMPT,
)
from .store import save_state


class AdaptivePMEngine:
    """A clean adaptive-discovery pipeline.

    Python owns provenance, state transitions, IDs, persistence and score ordering.
    LLMs own semantic interpretation, grounding, canonicalization and PM judgment.
    """

    def __init__(self):
        self.capture_model = get_structured_model(
            call_name="adaptive_pm.capture", schema=TurnCapture, max_tokens=1800
        )
        self.grounding_model = get_structured_model(
            call_name="adaptive_pm.ground", schema=GroundingAudit, max_tokens=1500
        )
        self.reasoning_model = get_structured_model(
            call_name="adaptive_pm.reason", schema=ReasoningUpdate, max_tokens=3500
        )
        self.planning_model = get_structured_model(
            call_name="adaptive_pm.plan", schema=PlanningResult, max_tokens=3000
        )
        self.audit_model = get_structured_model(
            call_name="adaptive_pm.question_audit", schema=QuestionAudit, max_tokens=900
        )
        self.control_model = get_chat_model(call_name="adaptive_pm.control", max_tokens=500)

    @staticmethod
    def new_state(raw_idea: str) -> InterviewState:
        return InterviewState(raw_idea=raw_idea)

    @staticmethod
    def _invoke_structured(model, system_prompt: str, payload: dict, schema):
        result = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
        ])
        return result if isinstance(result, schema) else schema.model_validate(result)

    @staticmethod
    def _recover_evidence(evidence: str, text: str) -> str | None:
        """Deterministic provenance only; no semantic judgment."""
        if evidence in text:
            return evidence
        tokens = [re.escape(token) for token in evidence.split() if token]
        if not tokens:
            return None
        match = re.search(r"\s+".join(tokens), text)
        return match.group(0) if match else None

    @staticmethod
    def _observation_id(turn: int, statement: str, evidence: str) -> str:
        digest = hashlib.sha1(f"{turn}|{statement}|{evidence}".encode("utf-8")).hexdigest()[:16]
        return f"obs_{digest}"

    def _capture(self, state: InterviewState, user_message: str) -> TurnCapture:
        payload = {
            "current_founder_message": user_message,
            "previous_pm_question": state.last_question,
            "previous_decision_key": state.last_decision_key,
            "recent_conversation": state.recent_messages[-4:],
            "known_actor_entity_names": sorted({
                entity
                for item in state.knowledge.values()
                for entity in item.entities
            })[:80],
        }
        return self._invoke_structured(self.capture_model, CAPTURE_PROMPT, payload, TurnCapture)

    def _source_ground(self, capture: TurnCapture, user_message: str) -> TurnCapture:
        grounded = []
        for fact in capture.facts:
            evidence = self._recover_evidence(fact.evidence, user_message)
            if evidence is None:
                print(f"SOURCE GROUNDING DROPPED: {fact.id} | evidence={fact.evidence!r}")
                continue
            grounded.append(fact.model_copy(update={"evidence": evidence}))

        boundary = capture.boundary
        if boundary is not None:
            evidence = self._recover_evidence(boundary.evidence, user_message)
            if evidence is None:
                print(f"SOURCE GROUNDING DROPPED BOUNDARY: {boundary.kind} | evidence={boundary.evidence!r}")
                boundary = None
            else:
                boundary = boundary.model_copy(update={"evidence": evidence})
        return capture.model_copy(update={"facts": grounded, "boundary": boundary})

    def _semantic_ground(
        self,
        state: InterviewState,
        capture: TurnCapture,
        user_message: str,
    ) -> list[Observation]:
        if not capture.facts:
            return []
        payload = {
            "current_founder_message": user_message,
            "previous_pm_question": state.last_question,
            "facts": [fact.model_dump(mode="json") for fact in capture.facts],
        }
        audit = self._invoke_structured(
            self.grounding_model, GROUNDING_PROMPT, payload, GroundingAudit
        )
        verdicts = {item.fact_id: item for item in audit.verdicts}
        observations: list[Observation] = []
        for fact in capture.facts:
            verdict = verdicts.get(fact.id)
            if verdict is None or not verdict.supported:
                reason = verdict.reason if verdict else "grounding model omitted verdict"
                print(f"SEMANTIC GROUNDING REJECTED: {fact.id} | {reason}")
                continue
            observations.append(Observation(
                id=self._observation_id(state.turn_count, fact.statement, fact.evidence),
                statement=fact.statement,
                evidence=fact.evidence,
                source=FactSource.USER,
                source_turn=state.turn_count,
                confidence=fact.confidence,
                entities=fact.entities,
                negative=fact.negative,
            ))
        return observations

    @staticmethod
    def _boundary_from_capture(
        state: InterviewState,
        capture: TurnCapture,
    ) -> DiscoveryBoundary | None:
        if capture.boundary is None:
            return None
        raw = capture.boundary
        digest = hashlib.sha1(
            f"{state.turn_count}|{raw.kind}|{raw.subject}|{raw.evidence}".encode("utf-8")
        ).hexdigest()[:16]
        return DiscoveryBoundary(
            id=f"boundary_{digest}",
            kind=raw.kind,
            subject=raw.subject,
            instruction=raw.instruction,
            evidence=raw.evidence,
            source_turn=state.turn_count,
            decision_key=state.last_decision_key,
        )

    def _control_response(
        self,
        state: InterviewState,
        intent: ConversationIntent,
        user_message: str,
    ) -> str:
        result = self.control_model.invoke([
            SystemMessage(content=CONTROL_RESPONSE_PROMPT),
            HumanMessage(content=json.dumps({
                "intent": intent.value,
                "founder_message": user_message,
                "previous_question": state.last_question,
                "current_product_context": [
                    item.statement for item in list(state.knowledge.values())[-20:]
                ],
            }, ensure_ascii=False)),
        ])
        return result.content.strip()

    def _reason(
        self,
        state: InterviewState,
        observations: list[Observation],
        intent: ConversationIntent,
    ) -> ReasoningUpdate:
        if not observations:
            return ReasoningUpdate()
        payload = {
            "turn": state.turn_count,
            "conversation_intent": intent.value,
            "new_grounded_observations": [
                item.model_dump(mode="json") for item in observations
            ],
            "existing_state": state.compact_context(),
        }
        return self._invoke_structured(
            self.reasoning_model, REASONING_PROMPT, payload, ReasoningUpdate
        )

    @staticmethod
    def _apply_reasoning(state: InterviewState, update: ReasoningUpdate) -> None:
        observation_ids = {item.id for item in state.observations}

        for mutation in update.knowledge_mutations:
            valid_obs = [
                identity
                for identity in mutation.observation_ids
                if identity in observation_ids
            ]

            if mutation.action == "REJECT":
                existing = state.knowledge.get(mutation.key)
                if existing:
                    state.knowledge[mutation.key] = existing.model_copy(update={
                        "status": KnowledgeStatus.REJECTED,
                        "last_updated_turn": state.turn_count,
                    })
                continue

            status = (
                KnowledgeStatus.DEFERRED
                if mutation.action == "DEFER"
                else mutation.status
            )

            record = KnowledgeRecord(
                key=mutation.key,
                statement=mutation.statement,
                category=mutation.category,
                status=status,
                evidence_observation_ids=valid_obs,
                entities=mutation.entities,
                affected_requirement_ids=mutation.affected_requirement_ids,
                first_seen_turn=(
                    state.knowledge[mutation.key].first_seen_turn
                    if mutation.key in state.knowledge
                    else state.turn_count
                ),
                last_updated_turn=state.turn_count,
            )

            if mutation.replaces_key and mutation.replaces_key != mutation.key:
                old = state.knowledge.get(mutation.replaces_key)
                if old:
                    state.knowledge[mutation.replaces_key] = old.model_copy(update={
                        "status": KnowledgeStatus.REJECTED,
                        "last_updated_turn": state.turn_count,
                    })

            state.knowledge[mutation.key] = record

        for concept in update.concepts:
            if concept.evidence_observation_ids:
                concept = concept.model_copy(update={
                    "evidence_observation_ids": [
                        identity
                        for identity in concept.evidence_observation_ids
                        if identity in observation_ids
                    ]
                })
            state.concepts[concept.id] = concept

        for requirement in update.requirement_updates:
            state.requirements[requirement.id] = RequirementRecord(
                **requirement.model_dump()
            )

        for implication in update.implications:
            implication = implication.model_copy(
                update={"status": KnowledgeStatus.PROPOSED}
            )
            state.implications[implication.id] = implication

        for contradiction in update.contradictions:
            state.contradictions[contradiction.id] = contradiction

        for key in update.resolved_decision_keys:
            if key in state.decisions:
                record = state.decisions[key]
                state.decisions[key] = record.model_copy(
                    update={"resolution": "ANSWERED"}
                )

    @staticmethod
    def _mark_previous_decision_from_control(
        state: InterviewState,
        intent: ConversationIntent,
    ) -> None:
        key = state.last_decision_key
        if not key or key not in state.decisions:
            return

        record = state.decisions[key]
        if intent in (
            ConversationIntent.OBJECTION,
            ConversationIntent.DESIGN_DEFERRAL,
        ):
            resolution = "REJECTED"
        elif intent == ConversationIntent.UNCERTAINTY:
            resolution = "DEFERRED"
        else:
            return

        state.decisions[key] = record.model_copy(update={"resolution": resolution})

    def _plan(
        self,
        state: InterviewState,
        intent: ConversationIntent,
    ) -> PlanningResult:
        payload = {
            "turn": state.turn_count,
            "founder_requested_advice": intent == ConversationIntent.ADVICE_REQUEST,
            "state": state.compact_context(),
            "completion_criteria": [
                "core product model coherent",
                "major entities and relationships understood",
                "primary workflows sufficiently specified",
                "high-impact business rules known",
                "architecture-changing uncertainties resolved or explicitly deferred",
                "major financial/inventory/security/integration decisions understood",
                "important lifecycle behavior known",
                "major failure paths and edge cases addressed",
                "MVP boundaries clear",
                "remaining unknowns low-impact or intentionally deferred",
            ],
        }
        return self._invoke_structured(
            self.planning_model, PLANNING_PROMPT, payload, PlanningResult
        )

    @staticmethod
    def _eligible_candidates(
        plan: PlanningResult,
        state: InterviewState,
    ) -> list[QuestionCandidate]:
        result: list[QuestionCandidate] = []
        for candidate in plan.candidates:
            if not candidate.eligible:
                continue
            prior = state.decisions.get(candidate.decision_key)
            if prior and prior.resolution in ("ANSWERED", "DEFERRED", "REJECTED"):
                continue
            result.append(candidate)
        return sorted(result, key=lambda item: item.score(), reverse=True)

    def _audit_question(
        self,
        state: InterviewState,
        candidate: QuestionCandidate,
    ) -> QuestionAudit:
        payload = {
            "candidate": candidate.model_dump(mode="json"),
            "question": candidate.question,
            "knowledge": [
                item.model_dump(mode="json")
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
        return self._invoke_structured(
            self.audit_model, QUESTION_AUDIT_PROMPT, payload, QuestionAudit
        )

    def _select_question(
        self,
        state: InterviewState,
        plan: PlanningResult,
    ) -> tuple[QuestionCandidate, str] | None:
        for candidate in self._eligible_candidates(plan, state)[:3]:
            audit = self._audit_question(state, candidate)

            if audit.reject_candidate:
                continue

            if audit.passed:
                return candidate, candidate.question.strip()

            if audit.revised_question and all([
                audit.atomic,
                audit.relevant,
                audit.non_repetitive,
                audit.respects_boundaries,
                not audit.implementation_detail,
            ]):
                return candidate, audit.revised_question.strip()

        return None

    @staticmethod
    def _record_message(
        state: InterviewState,
        speaker: str,
        text: str,
    ) -> None:
        state.recent_messages.append({
            "speaker": speaker,
            "text": text,
            "turn": state.turn_count,
        })
        state.recent_messages = state.recent_messages[-12:]

    def process_turn(
        self,
        state: InterviewState,
        user_message: str,
    ) -> str:
        """Process one founder turn and return the PM response/question."""
        state.turn_count += 1
        self._record_message(state, "founder", user_message)

        capture = self._source_ground(
            self._capture(state, user_message),
            user_message,
        )

        boundary = self._boundary_from_capture(state, capture)
        if boundary is not None and all(
            existing.id != boundary.id for existing in state.boundaries
        ):
            state.boundaries.append(boundary)

        if capture.intent in (
            ConversationIntent.CLARIFICATION,
            ConversationIntent.RATIONALE_REQUEST,
        ):
            response = self._control_response(
                state,
                capture.intent,
                user_message,
            )
            self._record_message(state, "pm", response)
            save_state(state)
            return response

        self._mark_previous_decision_from_control(state, capture.intent)

        if (
            capture.confirms_previous_answer
            and state.last_decision_key in state.decisions
        ):
            prior = state.decisions[state.last_decision_key]
            state.decisions[state.last_decision_key] = prior.model_copy(update={
                "resolution": "ANSWERED",
                "answer_summary": user_message.strip(),
            })

        observations = self._semantic_ground(
            state,
            capture,
            user_message,
        )

        existing_obs = {item.id for item in state.observations}
        state.observations.extend(
            item for item in observations if item.id not in existing_obs
        )

        reasoning = self._reason(
            state,
            observations,
            capture.intent,
        )
        self._apply_reasoning(state, reasoning)

        plan = self._plan(state, capture.intent)

        if plan.completion.complete:
            state.complete = True
            response = (
                "Discovery is sufficiently complete for the current product scope."
            )
            self._record_message(state, "pm", response)
            save_state(state)
            return response

        selected = self._select_question(state, plan)

        if selected is None:
            response = (
                "I have enough on this area for now. "
                "Tell me the next product decision you want to work through."
            )
            self._record_message(state, "pm", response)
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

        if plan.advice_options:
            advice = "\n".join(
                f"- {item}" for item in plan.advice_options[:3]
            )
            response = f"{advice}\n\n{question}"
        else:
            response = question

        self._record_message(state, "pm", response)
        save_state(state)
        return response

    def start(self, raw_idea: str) -> tuple[InterviewState, str]:
        state = self.new_state(raw_idea)
        save_state(state)
        response = self.process_turn(state, raw_idea)
        return state, response
