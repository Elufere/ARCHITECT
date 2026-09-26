from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .context import product_context
from .llm import structured_call
from .models import (
    ClassificationBatch,
    CompletionAssessment,
    ContradictionBatch,
    ContradictionRecord,
    ControlReply,
    DiscoveryBoundary,
    DiscoveryState,
    FactClassification,
    FactRecord,
    FactRelation,
    GroundingBatch,
    ImplicationBatch,
    ImplicationRecord,
    KnowledgeStatus,
    QuestionAuditBatch,
    QuestionCandidate,
    QuestionCandidateBatch,
    QuestionHistoryItem,
    ReconciliationBatch,
    RequirementActivationBatch,
    RequirementAssessmentBatch,
    RequirementRecord,
    SourceTurn,
    TurnIntent,
    TurnInterpretation,
)
from .prompts_capture import CLASSIFICATION, GROUNDING, RECONCILIATION, TURN_INTERPRETER
from .prompts_questions import COMPLETION, CONTROL_RESPONSE, QUESTION_AUDIT, QUESTION_CANDIDATES
from .prompts_reasoning import CONTRADICTIONS, IMPLICATIONS, REQUIREMENT_ACTIVATION, REQUIREMENT_ASSESSMENT
from .provenance import provenance_valid


@dataclass
class TurnResult:
    response: str
    state: DiscoveryState


class PMDiscoveryEngine:
    """Adaptive discovery engine built from the clean PM specification.

    Python owns orchestration, IDs, persistence-friendly state, provenance and
    deterministic ranking. LLM calls own semantic interpretation.
    """

    def process(self, state: DiscoveryState, user_message: str) -> TurnResult:
        turn = SourceTurn(number=state.turn_count, text=user_message)
        state.source_turns.append(turn)
        if not state.raw_idea:
            state.raw_idea = user_message

        interpretation = self._interpret_turn(state, turn)
        self._mark_previous_question_answered(state, interpretation)

        if interpretation.intent in {
            TurnIntent.CLARIFICATION,
            TurnIntent.RATIONALE_REQUEST,
            TurnIntent.SUMMARY_REQUEST,
            TurnIntent.ADVICE_REQUEST,
        }:
            response = self._control_response(state, turn, interpretation.intent)
            if (
                interpretation.intent
                in {TurnIntent.CLARIFICATION, TurnIntent.RATIONALE_REQUEST, TurnIntent.ADVICE_REQUEST}
                and response.rstrip().endswith("?")
            ):
                state.pending_question = response.strip()
            state.turn_count += 1
            return TurnResult(response=response, state=state)

        if interpretation.intent in {
            TurnIntent.DESIGN_DEFERRAL,
            TurnIntent.OBJECTION,
            TurnIntent.UNCERTAINTY,
        }:
            self._record_boundary(state, turn, interpretation)
            if interpretation.intent == TurnIntent.UNCERTAINTY:
                self._defer_pending_requirements(state)
            response = self._next_question_or_complete(state)
            state.turn_count += 1
            return TurnResult(response=response, state=state)

        new_fact_ids = self._capture_ground_classify_reconcile(
            state, turn, interpretation
        )

        if new_fact_ids:
            confirmed_new_fact_ids = [
                identity
                for identity in new_fact_ids
                if (
                    (fact := self._fact_by_id(state, identity)) is not None
                    and fact.status == KnowledgeStatus.CONFIRMED
                )
            ]
            if confirmed_new_fact_ids:
                self._derive_implications(state, confirmed_new_fact_ids)
                self._detect_contradictions(state, confirmed_new_fact_ids)
            self._activate_requirements(state, new_fact_ids)
            self._assess_requirements(state)

        response = self._next_question_or_complete(state)
        state.turn_count += 1
        return TurnResult(response=response, state=state)

    # ------------------------------------------------------------------
    # Turn interpretation and control
    # ------------------------------------------------------------------

    def _interpret_turn(
        self, state: DiscoveryState, turn: SourceTurn
    ) -> TurnInterpretation:
        return structured_call(
            call_name="turn_interpretation",
            schema=TurnInterpretation,
            instruction=TURN_INTERPRETER,
            payload={
                "latest_user_message": turn.text,
                "previous_pm_question": state.pending_question,
                "pending_decision_key": state.pending_decision_key,
                "recent_context": {
                    "confirmed_facts": product_context(state)["confirmed_facts"],
                    "recent_user_turns": product_context(state)["recent_user_turns"],
                },
            },
            max_tokens=1800,
        )

    def _control_response(
        self, state: DiscoveryState, turn: SourceTurn, intent: TurnIntent
    ) -> str:
        result = structured_call(
            call_name="control_response",
            schema=ControlReply,
            instruction=CONTROL_RESPONSE,
            payload={
                "intent": intent.value,
                "latest_user_message": turn.text,
                "previous_pm_question": state.pending_question,
                "context": product_context(state),
            },
            max_tokens=700,
        )
        return result.response.strip()

    def _mark_previous_question_answered(
        self, state: DiscoveryState, interpretation: TurnInterpretation
    ) -> None:
        if not state.question_history:
            return
        previous = state.question_history[-1]
        if previous.status != "ASKED":
            return
        intent = interpretation.intent
        if interpretation.answers_previous_question:
            previous.status = "ANSWERED"
        elif intent == TurnIntent.UNCERTAINTY:
            previous.status = "DEFERRED"
        elif intent in {TurnIntent.DESIGN_DEFERRAL, TurnIntent.OBJECTION}:
            previous.status = "REJECTED"

    def _record_boundary(
        self,
        state: DiscoveryState,
        turn: SourceTurn,
        interpretation: TurnInterpretation,
    ) -> None:
        if interpretation.intent == TurnIntent.DESIGN_DEFERRAL:
            kind = "DESIGN_DEFERRAL"
            instruction = (
                "Do not ask the founder for UI, navigation, interface, visual, or "
                "product-design implementation detail covered by this rejected inquiry. "
                "Only revisit if a distinct product rule genuinely cannot be decided without it."
            )
        elif interpretation.intent == TurnIntent.OBJECTION:
            kind = "REJECTED_INQUIRY"
            instruction = (
                "Do not repeat, paraphrase, or deepen the rejected inquiry. Move to a "
                "materially different high-value product decision."
            )
        else:
            kind = "DEFERRED_DECISION"
            instruction = (
                "The founder has not decided this yet. Treat it as deferred and do not "
                "keep pressing the same decision unless a later dependency makes it blocking."
            )

        state.boundaries.append(
            DiscoveryBoundary(
                kind=kind,
                source_turn_id=turn.id,
                evidence=turn.text,
                decision_key=state.pending_decision_key,
                question=state.pending_question,
                instruction=interpretation.boundary_summary or instruction,
            )
        )

    def _defer_pending_requirements(self, state: DiscoveryState) -> None:
        for key in state.pending_requirement_keys:
            requirement = state.requirements.get(key)
            if requirement is not None:
                requirement.status = KnowledgeStatus.DEFERRED

    # ------------------------------------------------------------------
    # Fact pipeline
    # ------------------------------------------------------------------

    def _capture_ground_classify_reconcile(
        self,
        state: DiscoveryState,
        turn: SourceTurn,
        interpretation: TurnInterpretation,
    ) -> list[str]:
        candidates = [
            item
            for item in interpretation.facts
            if provenance_valid(turn.text, item.evidence)
        ]
        if not candidates:
            return []

        grounding = structured_call(
            call_name="grounding",
            schema=GroundingBatch,
            instruction=GROUNDING,
            payload={
                "latest_user_message": turn.text,
                "previous_pm_question": state.pending_question,
                "candidates": [item.model_dump(mode="json") for item in candidates],
            },
            max_tokens=1800,
        )
        candidate_by_id = {item.id: item for item in candidates}
        grounded = []
        for verdict in grounding.verdicts:
            candidate = candidate_by_id.get(verdict.candidate_id)
            if candidate is None or not verdict.supported:
                continue
            grounded.append(
                {
                    "candidate_id": candidate.id,
                    "statement": verdict.normalized_statement.strip()
                    or candidate.statement,
                    "evidence": candidate.evidence,
                    "status": candidate.status.value,
                    "confidence": candidate.confidence,
                    "negative": candidate.negative,
                }
            )
        if not grounded:
            return []

        classifications = structured_call(
            call_name="classification",
            schema=ClassificationBatch,
            instruction=CLASSIFICATION,
            payload={
                "grounded_facts": grounded,
                "existing_canonical_keys": [
                    fact.canonical_key for fact in state.facts if fact.active
                ],
            },
            max_tokens=1800,
        )
        classification_by_id = {
            item.candidate_id: item for item in classifications.items
        }

        classified = []
        for item in grounded:
            classification = classification_by_id.get(item["candidate_id"])
            if classification is None:
                continue
            classified.append(
                {
                    **item,
                    "canonical_key": classification.canonical_key,
                    "domains": classification.domains,
                    "entities": classification.entities,
                    "summary": classification.summary,
                }
            )
        if not classified:
            return []

        reconciliation = structured_call(
            call_name="reconciliation",
            schema=ReconciliationBatch,
            instruction=RECONCILIATION,
            payload={
                "turn_intent": interpretation.intent.value,
                "new_facts": classified,
                "existing_facts": [
                    {
                        "id": fact.id,
                        "canonical_key": fact.canonical_key,
                        "statement": fact.statement,
                        "domains": fact.domains,
                        "entities": fact.entities,
                        "negative": fact.negative,
                        "status": fact.status.value,
                    }
                    for fact in state.facts
                    if fact.active and fact.status in {
                        KnowledgeStatus.CONFIRMED,
                        KnowledgeStatus.PROPOSED,
                    }
                ],
            },
            max_tokens=2200,
        )
        decision_by_id = {
            item.candidate_id: item for item in reconciliation.decisions
        }

        committed: list[str] = []
        for item in classified:
            decision = decision_by_id.get(item["candidate_id"])
            if decision is None:
                continue
            matched = self._fact_by_id(state, decision.matched_fact_id)

            if decision.relation == FactRelation.DUPLICATE and matched is not None:
                if turn.id not in matched.source_turn_ids:
                    matched.source_turn_ids.append(turn.id)
                if item["evidence"] not in matched.evidence:
                    matched.evidence.append(item["evidence"])
                continue

            new_fact = FactRecord(
                canonical_key=item["canonical_key"],
                statement=decision.canonical_statement.strip() or item["statement"],
                domains=item["domains"],
                entities=item["entities"],
                status=KnowledgeStatus(item["status"]),
                confidence=item["confidence"],
                negative=item["negative"],
                source_turn_ids=[turn.id],
                evidence=[item["evidence"]],
            )

            if (
                decision.relation in {FactRelation.REFINEMENT, FactRelation.CORRECTION}
                and matched is not None
            ):
                matched.active = False
                matched.superseded_by = new_fact.id
                if decision.relation == FactRelation.CORRECTION:
                    for conflict in state.contradictions:
                        if matched.id in conflict.fact_ids:
                            conflict.resolved = True

            state.facts.append(new_fact)
            committed.append(new_fact.id)

        return committed

    # ------------------------------------------------------------------
    # Product reasoning
    # ------------------------------------------------------------------

    def _derive_implications(
        self, state: DiscoveryState, new_fact_ids: list[str]
    ) -> None:
        result = structured_call(
            call_name="implications",
            schema=ImplicationBatch,
            instruction=IMPLICATIONS,
            payload={
                "new_fact_ids": new_fact_ids,
                "context": product_context(state),
            },
            max_tokens=1800,
        )
        existing = {item.statement.strip().lower() for item in state.implications}
        valid_fact_ids = {item.id for item in state.facts}
        for proposal in result.items:
            statement_key = proposal.statement.strip().lower()
            if not statement_key or statement_key in existing:
                continue
            basis = [
                identity
                for identity in proposal.based_on_fact_ids
                if identity in valid_fact_ids
            ]
            if not basis:
                continue
            record = ImplicationRecord(
                statement=proposal.statement,
                based_on_fact_ids=basis,
                domains=proposal.domains,
                needs_validation=proposal.needs_validation,
                importance=proposal.importance,
            )
            state.implications.append(record)
            existing.add(statement_key)

    def _activate_requirements(
        self, state: DiscoveryState, new_fact_ids: list[str]
    ) -> None:
        result = structured_call(
            call_name="requirement_activation",
            schema=RequirementActivationBatch,
            instruction=REQUIREMENT_ACTIVATION,
            payload={
                "new_fact_ids": new_fact_ids,
                "context": product_context(state),
            },
            max_tokens=2400,
        )
        valid_fact_ids = {item.id for item in state.facts}
        valid_implication_ids = {item.id for item in state.implications}

        for proposal in result.items:
            key = self._normalize_key(proposal.key)
            if not key:
                continue
            fact_ids = [
                identity
                for identity in proposal.basis_fact_ids
                if identity in valid_fact_ids
            ]
            implication_ids = [
                identity
                for identity in proposal.basis_implication_ids
                if identity in valid_implication_ids
            ]
            existing = state.requirements.get(key)
            if existing is None:
                confirmed_basis = any(
                    (fact := self._fact_by_id(state, identity)) is not None
                    and fact.status == KnowledgeStatus.CONFIRMED
                    for identity in fact_ids
                )
                status = (
                    KnowledgeStatus.UNKNOWN
                    if confirmed_basis
                    else KnowledgeStatus.PROPOSED
                )
                state.requirements[key] = RequirementRecord(
                    key=key,
                    label=proposal.label,
                    description=proposal.description,
                    status=status,
                    evidence_fact_ids=fact_ids,
                    implication_ids=implication_ids,
                    dependencies=[
                        self._normalize_key(item)
                        for item in proposal.dependencies
                        if self._normalize_key(item)
                    ],
                    business_impact=proposal.business_impact,
                    architecture_impact=proposal.architecture_impact,
                    risk=proposal.risk,
                    activation_reason=proposal.reason,
                )
                continue

            existing.evidence_fact_ids = self._unique(
                [*existing.evidence_fact_ids, *fact_ids]
            )
            existing.implication_ids = self._unique(
                [*existing.implication_ids, *implication_ids]
            )
            existing.dependencies = self._unique(
                [
                    *existing.dependencies,
                    *[
                        self._normalize_key(item)
                        for item in proposal.dependencies
                        if self._normalize_key(item)
                    ],
                ]
            )
            existing.business_impact = max(
                existing.business_impact, proposal.business_impact
            )
            existing.architecture_impact = max(
                existing.architecture_impact, proposal.architecture_impact
            )
            existing.risk = max(existing.risk, proposal.risk)

        # Safety net for dependency propagation: if the model references a
        # dependency but omits its own proposal, retain it as a PROPOSED
        # requirement so it cannot disappear from the requirement graph.
        referenced_dependencies = {
            dependency
            for requirement in state.requirements.values()
            for dependency in requirement.dependencies
            if dependency
        }
        for dependency in referenced_dependencies:
            if dependency in state.requirements:
                continue
            state.requirements[dependency] = RequirementRecord(
                key=dependency,
                label=dependency.replace(".", " ").replace("_", " ").title(),
                description=(
                    "Dependency activated by another product requirement; "
                    "requires validation before it can be treated as confirmed."
                ),
                status=KnowledgeStatus.PROPOSED,
                activation_reason="Dependency propagation",
            )

    def _assess_requirements(self, state: DiscoveryState) -> None:
        if not state.requirements:
            return
        result = structured_call(
            call_name="requirement_assessment",
            schema=RequirementAssessmentBatch,
            instruction=REQUIREMENT_ASSESSMENT,
            payload={"context": product_context(state)},
            max_tokens=3000,
        )
        valid_fact_ids = {item.id for item in state.facts if item.active}
        for assessment in result.items:
            requirement = state.requirements.get(
                self._normalize_key(assessment.key)
            )
            if requirement is None:
                continue
            requirement.status = assessment.status
            requirement.coverage = assessment.coverage
            requirement.depth = assessment.depth
            requirement.evidence_fact_ids = self._unique(
                [
                    *requirement.evidence_fact_ids,
                    *[
                        identity
                        for identity in assessment.supporting_fact_ids
                        if identity in valid_fact_ids
                    ],
                ]
            )
            requirement.missing_decisions = self._unique(
                assessment.missing_decisions
            )

    def _detect_contradictions(
        self, state: DiscoveryState, new_fact_ids: list[str]
    ) -> None:
        result = structured_call(
            call_name="contradictions",
            schema=ContradictionBatch,
            instruction=CONTRADICTIONS,
            payload={
                "new_fact_ids": new_fact_ids,
                "confirmed_facts": product_context(state)["confirmed_facts"],
            },
            max_tokens=1400,
        )
        valid_ids = {fact.id for fact in state.facts if fact.active}
        existing_sets = {
            frozenset(item.fact_ids)
            for item in state.contradictions
            if not item.resolved
        }
        for proposal in result.items:
            fact_ids = self._unique(
                [identity for identity in proposal.fact_ids if identity in valid_ids]
            )
            if len(fact_ids) < 2 or frozenset(fact_ids) in existing_sets:
                continue
            record = ContradictionRecord(
                fact_ids=fact_ids,
                issue=proposal.issue,
                blocking=proposal.blocking,
            )
            state.contradictions.append(record)
            existing_sets.add(frozenset(fact_ids))

    # ------------------------------------------------------------------
    # Question selection
    # ------------------------------------------------------------------

    def _next_question_or_complete(self, state: DiscoveryState) -> str:
        completion = structured_call(
            call_name="completion",
            schema=CompletionAssessment,
            instruction=COMPLETION,
            payload={"context": product_context(state)},
            max_tokens=1200,
        )
        if completion.complete:
            state.complete = True
            state.completion_reason = completion.reason
            state.pending_question = None
            state.pending_decision_key = None
            state.pending_requirement_keys = []
            return (
                "Discovery is complete enough for an implementation-ready PRD. "
                + completion.reason
            )

        first_candidates = self._generate_question_candidates(
            state, completion, repair_feedback=None
        )
        selected, audit_feedback = self._audit_and_select(
            state, first_candidates
        )

        if selected is None:
            # One semantic repair pass only. This is intentionally different
            # from blindly regenerating the same question until something passes.
            repaired_candidates = self._generate_question_candidates(
                state,
                completion,
                repair_feedback=audit_feedback,
            )
            selected, _ = self._audit_and_select(
                state, repaired_candidates
            )

        if selected is None:
            return self._no_question_fallback(state, completion)

        state.pending_question = selected.question.strip()
        state.pending_decision_key = selected.decision_key
        state.pending_requirement_keys = [
            key
            for key in selected.requirement_keys
            if key in state.requirements
        ]
        state.question_history.append(
            QuestionHistoryItem(
                turn=state.turn_count,
                decision_key=selected.decision_key,
                objective=selected.objective,
                question=selected.question.strip(),
                requirement_keys=state.pending_requirement_keys,
            )
        )
        return selected.question.strip()

    def _generate_question_candidates(
        self,
        state: DiscoveryState,
        completion: CompletionAssessment,
        repair_feedback: list[dict] | None,
    ) -> list[QuestionCandidate]:
        instruction = QUESTION_CANDIDATES
        if repair_feedback:
            instruction += (
                "\nThe previous candidate set failed quality review. Generate "
                "materially different candidates that address the audit reasons. "
                "Do not merely paraphrase rejected candidates. If no legitimate "
                "high-value question exists, return an empty list."
            )
        result = structured_call(
            call_name=(
                "question_candidates_repair"
                if repair_feedback
                else "question_candidates"
            ),
            schema=QuestionCandidateBatch,
            instruction=instruction,
            payload={
                "blocking_requirement_keys": completion.blocking_requirement_keys,
                "unresolved_high_impact": completion.unresolved_high_impact,
                "repair_feedback": repair_feedback or [],
                "context": product_context(state),
            },
            max_tokens=2800,
        )
        return result.items

    def _audit_and_select(
        self,
        state: DiscoveryState,
        candidates: list[QuestionCandidate],
    ) -> tuple[QuestionCandidate | None, list[dict]]:
        if not candidates:
            return None, []

        audit = structured_call(
            call_name="question_audit",
            schema=QuestionAuditBatch,
            instruction=QUESTION_AUDIT,
            payload={
                "candidates": [
                    item.model_dump(mode="json") for item in candidates
                ],
                "context": product_context(state),
            },
            max_tokens=1800,
        )
        audit_by_id = {item.candidate_id: item for item in audit.items}
        blocked_keys = self._blocked_decision_keys(state)
        eligible = [
            item
            for item in candidates
            if (audit_by_id.get(item.id) is not None)
            and audit_by_id[item.id].eligible
            and item.decision_key not in blocked_keys
        ]
        feedback = [
            {
                "candidate": item.model_dump(mode="json"),
                "audit": (
                    audit_by_id[item.id].model_dump(mode="json")
                    if item.id in audit_by_id
                    else {"eligible": False, "reason": "No audit verdict returned"}
                ),
            }
            for item in candidates
            if item not in eligible
        ]
        if not eligible:
            return None, feedback
        return max(eligible, key=self._priority_score), feedback

    def _no_question_fallback(
        self, state: DiscoveryState, completion: CompletionAssessment
    ) -> str:
        # Fail safe: never lower the quality bar simply to keep asking.
        state.pending_question = None
        state.pending_decision_key = None
        state.pending_requirement_keys = []

        unresolved_blockers = [
            req
            for req in state.requirements.values()
            if req.status in {KnowledgeStatus.UNKNOWN, KnowledgeStatus.PROPOSED}
            and req.missing_decisions
            and max(req.business_impact, req.architecture_impact, req.risk) >= 0.65
        ]
        open_blocking_conflicts = [
            conflict
            for conflict in state.contradictions
            if conflict.blocking and not conflict.resolved
        ]
        if not unresolved_blockers and not open_blocking_conflicts:
            state.complete = True
            state.completion_reason = (
                "No further high-value question survived the quality audit; "
                "remaining unknowns are low-value or deferred."
            )
            return (
                "Discovery is complete enough for an implementation-ready PRD. "
                + state.completion_reason
            )

        state.completion_reason = (
            "No candidate question passed the quality audit while material "
            "unknowns still remain. "
            + completion.reason
        )
        return (
            "I can't justify another question without lowering the discovery "
            "quality bar. "
            + completion.reason
        )

    @staticmethod
    def _priority_score(candidate: QuestionCandidate) -> float:
        return (
            candidate.business_impact
            + candidate.architecture_impact
            + candidate.dependency_unlock
            + candidate.uncertainty
            + candidate.risk
            + candidate.contextual_relevance
            - candidate.repetition_penalty
            - candidate.premature_detail_penalty
            - candidate.user_fatigue_penalty
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_key(value: str) -> str:
        return ".".join(
            part
            for part in "".join(
                char.lower() if char.isalnum() else "."
                for char in (value or "").strip()
            ).split(".")
            if part
        )[:120]

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    @staticmethod
    def _fact_by_id(
        state: DiscoveryState, identity: str | None
    ) -> FactRecord | None:
        if not identity:
            return None
        return next((item for item in state.facts if item.id == identity), None)

    @staticmethod
    def _blocked_decision_keys(state: DiscoveryState) -> set[str]:
        return {
            boundary.decision_key
            for boundary in state.boundaries
            if boundary.decision_key
            and boundary.kind in {
                "DESIGN_DEFERRAL",
                "REJECTED_INQUIRY",
                "DEFERRED_DECISION",
            }
        }
