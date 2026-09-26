from __future__ import annotations

import hashlib
import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm import get_chat_model, get_structured_model
from .models import (
    ConversationIntent,
    DiscoveryBoundary,
    FactSource,
    GroundingAudit,
    InterviewState,
    KnowledgeStatus,
    Observation,
    TurnCapture,
)
from .prompts import CAPTURE_PROMPT, CONTROL_RESPONSE_PROMPT, GROUNDING_PROMPT
from .runtime import invoke_structured, observation_id, previous_pm_response, recover_evidence


class CaptureStage:
    """Capture founder meaning before any product classification."""

    def __init__(self):
        self.capture_model = get_structured_model(
            call_name="adaptive_pm.capture",
            schema=TurnCapture,
            max_tokens=1800,
        )
        self.grounding_model = get_structured_model(
            call_name="adaptive_pm.ground",
            schema=GroundingAudit,
            max_tokens=1500,
        )
        self.control_model = get_chat_model(
            call_name="adaptive_pm.control",
            max_tokens=500,
        )

    def capture(self, state: InterviewState, user_message: str) -> TurnCapture:
        payload = {
            "current_founder_message": user_message,
            "previous_pm_question": state.last_question,
            "previous_pm_response": previous_pm_response(state),
            "previous_decision_key": state.last_decision_key,
            "proposed_recommendations": [
                item.model_dump(mode="json")
                for item in state.recommendations.values()
                if item.status == KnowledgeStatus.PROPOSED
            ],
            "recent_conversation": state.recent_messages[-4:],
            "known_actor_entity_names": sorted({
                entity
                for item in state.knowledge.values()
                for entity in item.entities
            })[:80],
        }
        result = invoke_structured(
            self.capture_model,
            CAPTURE_PROMPT,
            payload,
            TurnCapture,
        )
        return self.source_ground(result, user_message)

    @staticmethod
    def source_ground(capture: TurnCapture, user_message: str) -> TurnCapture:
        """Python verifies provenance; it does not decide semantic support."""
        grounded_facts = []
        for fact in capture.facts:
            evidence = recover_evidence(fact.evidence, user_message)
            if evidence is None:
                print(
                    f"SOURCE GROUNDING DROPPED: {fact.id} | "
                    f"evidence={fact.evidence!r}"
                )
                continue
            grounded_facts.append(
                fact.model_copy(update={"evidence": evidence})
            )

        boundary = capture.boundary
        if boundary is not None:
            evidence = recover_evidence(boundary.evidence, user_message)
            if evidence is None:
                print(
                    "SOURCE GROUNDING DROPPED BOUNDARY: "
                    f"{boundary.kind} | evidence={boundary.evidence!r}"
                )
                boundary = None
            else:
                boundary = boundary.model_copy(
                    update={"evidence": evidence}
                )

        return capture.model_copy(
            update={"facts": grounded_facts, "boundary": boundary}
        )

    def semantic_ground(
        self,
        state: InterviewState,
        capture: TurnCapture,
        user_message: str,
    ) -> list[Observation]:
        """The LLM decides whether grounded text supports the proposed meaning."""
        if not capture.facts:
            return []

        payload = {
            "current_founder_message": user_message,
            "previous_pm_question": state.last_question,
            "previous_pm_response": previous_pm_response(state),
            "proposed_recommendations": [
                item.model_dump(mode="json")
                for item in state.recommendations.values()
                if item.status == KnowledgeStatus.PROPOSED
            ],
            "facts": [
                item.model_dump(mode="json")
                for item in capture.facts
            ],
        }
        audit = invoke_structured(
            self.grounding_model,
            GROUNDING_PROMPT,
            payload,
            GroundingAudit,
        )
        verdicts = {item.fact_id: item for item in audit.verdicts}
        observations: list[Observation] = []

        for fact in capture.facts:
            verdict = verdicts.get(fact.id)
            if verdict is None or not verdict.supported:
                reason = (
                    verdict.reason
                    if verdict
                    else "grounding model omitted verdict"
                )
                print(
                    f"SEMANTIC GROUNDING REJECTED: {fact.id} | {reason}"
                )
                continue

            observations.append(Observation(
                id=observation_id(
                    state.turn_count,
                    fact.statement,
                    fact.evidence,
                ),
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
    def boundary_from_capture(
        state: InterviewState,
        capture: TurnCapture,
    ) -> DiscoveryBoundary | None:
        if capture.boundary is None:
            return None
        raw = capture.boundary
        digest = hashlib.sha1(
            (
                f"{state.turn_count}|{raw.kind}|"
                f"{raw.subject}|{raw.evidence}"
            ).encode("utf-8")
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

    def control_response(
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
                    item.statement
                    for item in list(state.knowledge.values())[-20:]
                ],
            }, ensure_ascii=False)),
        ])
        return result.content.strip()
