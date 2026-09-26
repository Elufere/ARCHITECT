from __future__ import annotations

from agents.llm import get_structured_model
from .models import (
    CanonicalizationResult,
    ConversationIntent,
    InterviewState,
    KnowledgeRecord,
    KnowledgeStatus,
    Observation,
)
from .runtime import invoke_structured
from .stage_prompts import CANONICALIZE_PROMPT


class KnowledgeStage:
    """Canonicalize grounded observations without planning questions."""

    def __init__(self):
        self.model = get_structured_model(
            call_name="adaptive_pm.canonicalize",
            schema=CanonicalizationResult,
            max_tokens=2400,
        )

    def canonicalize(
        self,
        state: InterviewState,
        observations: list[Observation],
        intent: ConversationIntent,
    ) -> CanonicalizationResult:
        if not observations:
            return CanonicalizationResult()

        payload = {
            "turn": state.turn_count,
            "conversation_intent": intent.value,
            "new_grounded_observations": [
                item.model_dump(mode="json")
                for item in observations
            ],
            "existing_knowledge": [
                item.model_dump(mode="json")
                for item in state.knowledge.values()
            ],
            "product_concepts": [
                item.model_dump(mode="json")
                for item in state.concepts.values()
            ],
            "decision_history": [
                item.model_dump(mode="json")
                for item in state.decisions.values()
            ],
        }
        return invoke_structured(
            self.model,
            CANONICALIZE_PROMPT,
            payload,
            CanonicalizationResult,
        )

    @staticmethod
    def apply(
        state: InterviewState,
        result: CanonicalizationResult,
    ) -> list[str]:
        observation_ids = {item.id for item in state.observations}
        changed_keys: list[str] = []

        for mutation in result.knowledge_mutations:
            valid_obs = [
                identity
                for identity in mutation.observation_ids
                if identity in observation_ids
            ]
            existing = state.knowledge.get(mutation.key)

            if mutation.action == "REJECT":
                if existing:
                    state.knowledge_history.append(
                        existing.model_copy(update={
                            "status": KnowledgeStatus.REJECTED,
                            "last_updated_turn": state.turn_count,
                        })
                    )
                    state.knowledge.pop(mutation.key, None)
                    changed_keys.append(mutation.key)
                continue

            status = (
                KnowledgeStatus.DEFERRED
                if mutation.action == "DEFER"
                else mutation.status
            )

            # Canonical product knowledge, including tentative/proposed
            # founder statements, must trace to a grounded observation.
            if not valid_obs:
                print(
                    "KNOWLEDGE MUTATION DROPPED "
                    f"(no grounded evidence): {mutation.action} "
                    f"{mutation.key}"
                )
                continue

            evidence_ids = list(valid_obs)
            entities = list(mutation.entities)
            requirement_ids = list(
                mutation.affected_requirement_ids
            )

            if existing and mutation.action in ("ADD", "REFINE"):
                evidence_ids = list(dict.fromkeys([
                    *existing.evidence_observation_ids,
                    *evidence_ids,
                ]))
                entities = list(dict.fromkeys([
                    *existing.entities,
                    *entities,
                ]))
                requirement_ids = list(dict.fromkeys([
                    *existing.affected_requirement_ids,
                    *requirement_ids,
                ]))

            if existing and mutation.action == "SUPERSEDE":
                state.knowledge_history.append(
                    existing.model_copy(update={
                        "status": KnowledgeStatus.REJECTED,
                        "last_updated_turn": state.turn_count,
                    })
                )

            record = KnowledgeRecord(
                key=mutation.key,
                statement=mutation.statement,
                category=mutation.category,
                status=status,
                evidence_observation_ids=evidence_ids,
                entities=entities,
                affected_requirement_ids=requirement_ids,
                first_seen_turn=(
                    existing.first_seen_turn
                    if existing
                    else state.turn_count
                ),
                last_updated_turn=state.turn_count,
            )

            if (
                mutation.replaces_key
                and mutation.replaces_key != mutation.key
            ):
                replaced = state.knowledge.pop(
                    mutation.replaces_key,
                    None,
                )
                if replaced:
                    state.knowledge_history.append(
                        replaced.model_copy(update={
                            "status": KnowledgeStatus.REJECTED,
                            "last_updated_turn": state.turn_count,
                        })
                    )
                    changed_keys.append(mutation.replaces_key)

            state.knowledge[mutation.key] = record
            changed_keys.append(mutation.key)

        for concept in result.concepts:
            valid_evidence = [
                identity
                for identity in concept.evidence_observation_ids
                if identity in observation_ids
            ]
            if not valid_evidence:
                print(
                    "PRODUCT CONCEPT DROPPED "
                    f"(no grounded evidence): {concept.id}"
                )
                continue
            state.concepts[concept.id] = concept.model_copy(
                update={
                    "evidence_observation_ids": valid_evidence
                }
            )

        for decision_key in result.resolved_decision_keys:
            if decision_key in state.decisions:
                prior = state.decisions[decision_key]
                state.decisions[decision_key] = prior.model_copy(
                    update={"resolution": "ANSWERED"}
                )

        return list(dict.fromkeys(changed_keys))
