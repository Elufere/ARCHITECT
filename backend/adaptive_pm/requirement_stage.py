from __future__ import annotations

from agents.llm import get_structured_model
from .models import (
    ContradictionResult,
    ImplicationResult,
    InterviewState,
    KnowledgeStatus,
    RequirementReasoningResult,
    RequirementRecord,
)
from .runtime import invoke_structured
from .stage_prompts import (
    CONTRADICTION_PROMPT,
    IMPLICATION_PROMPT,
    REQUIREMENT_PROMPT,
)


class RequirementStage:
    """Reason about dynamic requirements after canonical knowledge changes."""

    def __init__(self):
        self.model = get_structured_model(
            call_name="adaptive_pm.requirements",
            schema=RequirementReasoningResult,
            max_tokens=3000,
        )

    def reason(
        self,
        state: InterviewState,
        changed_keys: list[str],
    ) -> RequirementReasoningResult:
        if not changed_keys:
            return RequirementReasoningResult()

        payload = {
            "turn": state.turn_count,
            "changed_knowledge_keys": changed_keys,
            "canonical_knowledge": [
                item.model_dump(mode="json")
                for item in state.knowledge.values()
            ],
            "product_concepts": [
                item.model_dump(mode="json")
                for item in state.concepts.values()
            ],
            "existing_requirements": [
                item.model_dump(mode="json")
                for item in state.requirements.values()
            ],
        }
        return invoke_structured(
            self.model,
            REQUIREMENT_PROMPT,
            payload,
            RequirementReasoningResult,
        )

    @staticmethod
    def apply(
        state: InterviewState,
        result: RequirementReasoningResult,
    ) -> None:
        for identity in result.deactivate_requirement_ids:
            state.requirements.pop(identity, None)

        knowledge_keys = set(state.knowledge)
        touched_by_requirement: dict[str, set[str]] = {}

        for mutation in result.requirement_updates:
            payload = mutation.model_dump()
            payload["evidence_knowledge_keys"] = [
                key
                for key in mutation.evidence_knowledge_keys
                if key in knowledge_keys
            ]
            record = RequirementRecord(**payload)
            state.requirements[record.id] = record
            for key in record.evidence_knowledge_keys:
                touched_by_requirement.setdefault(key, set()).add(
                    record.id
                )

        # Keep reverse links in canonical knowledge so one founder fact can
        # support multiple requirements without being duplicated.
        for key, requirement_ids in touched_by_requirement.items():
            record = state.knowledge.get(key)
            if record is None:
                continue
            state.knowledge[key] = record.model_copy(update={
                "affected_requirement_ids": list(dict.fromkeys([
                    *record.affected_requirement_ids,
                    *sorted(requirement_ids),
                ]))
            })


class ImplicationStage:
    """Derive proposed implications without promoting them to facts."""

    def __init__(self):
        self.model = get_structured_model(
            call_name="adaptive_pm.implications",
            schema=ImplicationResult,
            max_tokens=1800,
        )

    def derive(
        self,
        state: InterviewState,
        changed_keys: list[str],
    ) -> ImplicationResult:
        if not changed_keys:
            return ImplicationResult()

        payload = {
            "turn": state.turn_count,
            "changed_knowledge_keys": changed_keys,
            "canonical_knowledge": [
                item.model_dump(mode="json")
                for item in state.knowledge.values()
            ],
            "requirements": [
                item.model_dump(mode="json")
                for item in state.requirements.values()
            ],
            "existing_implications": [
                item.model_dump(mode="json")
                for item in state.implications.values()
            ],
        }
        return invoke_structured(
            self.model,
            IMPLICATION_PROMPT,
            payload,
            ImplicationResult,
        )

    @staticmethod
    def apply(
        state: InterviewState,
        result: ImplicationResult,
    ) -> None:
        knowledge_keys = set(state.knowledge)

        # Remove implications whose source knowledge was superseded/rejected.
        state.implications = {
            identity: item
            for identity, item in state.implications.items()
            if any(
                key in knowledge_keys
                for key in item.source_knowledge_keys
            )
        }

        for implication in result.implications:
            valid_sources = [
                key
                for key in implication.source_knowledge_keys
                if key in knowledge_keys
            ]
            if not valid_sources:
                print(
                    "IMPLICATION DROPPED "
                    f"(no canonical source): {implication.id}"
                )
                continue
            state.implications[implication.id] = (
                implication.model_copy(update={
                    "status": KnowledgeStatus.PROPOSED,
                    "source_knowledge_keys": valid_sources,
                })
            )


class ContradictionStage:
    """Compare current canonical facts after updates."""

    def __init__(self):
        self.model = get_structured_model(
            call_name="adaptive_pm.contradictions",
            schema=ContradictionResult,
            max_tokens=1500,
        )

    def detect(
        self,
        state: InterviewState,
        changed_keys: list[str],
    ) -> ContradictionResult:
        confirmed = [
            item
            for item in state.knowledge.values()
            if item.status == KnowledgeStatus.CONFIRMED
        ]
        if not changed_keys or len(confirmed) < 2:
            return ContradictionResult()

        payload = {
            "turn": state.turn_count,
            "changed_knowledge_keys": changed_keys,
            "confirmed_knowledge": [
                item.model_dump(mode="json")
                for item in confirmed
            ],
            "open_contradictions": [
                item.model_dump(mode="json")
                for item in state.contradictions.values()
                if not item.resolved
            ],
        }
        return invoke_structured(
            self.model,
            CONTRADICTION_PROMPT,
            payload,
            ContradictionResult,
        )

    @staticmethod
    def apply(
        state: InterviewState,
        result: ContradictionResult,
    ) -> None:
        for identity in result.resolved_contradiction_ids:
            existing = state.contradictions.get(identity)
            if existing:
                state.contradictions[identity] = (
                    existing.model_copy(update={"resolved": True})
                )

        current_keys = set(state.knowledge)
        for contradiction in result.contradictions:
            valid_keys = [
                key
                for key in contradiction.knowledge_keys
                if key in current_keys
            ]
            if len(valid_keys) < 2:
                continue
            state.contradictions[contradiction.id] = (
                contradiction.model_copy(
                    update={"knowledge_keys": valid_keys}
                )
            )
