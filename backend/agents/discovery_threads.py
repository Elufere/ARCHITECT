"""Conversation-level discovery threads for coherent product interviews.

Threads are not product facts and do not replace requirements. They control
conversation trajectory: stay with the part of the product currently being
understood, follow causal consequences of confirmed decisions, and defer
unrelated requirements until their thread becomes relevant.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Dict, List, Optional, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator, model_validator

from agents.discovery_coverage import fact_id
from agents.discovery_deferrals import (
    DeferralKind,
    FreeTextDeferralReview,
    apply_deferral,
)
from agents.llm import get_structured_model
from agents.llm_errors import ExtractionFailed, raise_if_llm_failure
from agents.state import AgentState, DiscoveryScope, DiscoveryTopic, KnowledgeState, TOPIC_KEY_MAP
from agents.product_concepts import ProductConcept, ProductConceptKind
from agents.external_systems import ExternalSystem
from agents.conversation_language import message_text


class ThreadStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETE = "COMPLETE"


class DiscoveryThread(BaseModel):
    id: str
    label: str
    objective: str
    scope: DiscoveryScope
    parent_thread_id: Optional[str] = None
    status: ThreadStatus = ThreadStatus.ACTIVE
    trigger_fact_ids: List[str] = Field(default_factory=list)
    last_active_turn: int = 0


class ThreadFrontierInquiry(BaseModel):
    decision_key: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{1,79}$")
    topic: DiscoveryTopic
    anchor_gap: Optional[str] = None
    objective: str
    question_hint: str
    reason: str
    related_fact_ids: List[str] = Field(default_factory=list)
    information_gain: float = Field(default=0.8, ge=0, le=1)
    causal_relevance: float = Field(default=1.0, ge=0, le=1)
    conversation_continuity: float = Field(default=1.0, ge=0, le=1)
    architecture_impact: float = Field(default=0.6, ge=0, le=1)
    business_risk: float = Field(default=0.5, ge=0, le=1)
    question_cost: float = Field(default=0.0, ge=0, le=1)

    @field_validator("decision_key", mode="before")
    @classmethod
    def normalize_decision_key(cls, value):
        if not isinstance(value, str):
            return value
        normalized = re.sub(r"[^a-z0-9_.-]+", "_", value.strip().lower()).strip("_.-")
        return normalized[:80]

    @model_validator(mode="after")
    def valid_anchor(self):
        if self.anchor_gap:
            key = self.anchor_gap.split("::", 1)[0]
            if key not in TOPIC_KEY_MAP[self.topic]:
                # anchor_gap is normalization metadata only. A useful thread
                # decision must never be rejected merely because the model
                # selected an imperfect legacy storage anchor.
                self.anchor_gap = None
        return self


class ThreadFeedbackKind(str, Enum):
    QUESTION_TOO_BROAD = "QUESTION_TOO_BROAD"
    IMPLEMENTATION_DEFERRED = "IMPLEMENTATION_DEFERRED"
    DECISION_DEFERRED = "DECISION_DEFERRED"
    PRODUCT_SCOPE_CLOSED = "PRODUCT_SCOPE_CLOSED"


class ThreadFeedback(BaseModel):
    kind: ThreadFeedbackKind
    evidence: str
    instruction: str


class DiscoveryThreadPlan(BaseModel):
    thread_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{1,79}$")
    thread_label: str
    thread_objective: str
    parent_thread_id: Optional[str] = None
    frontier: Optional[ThreadFrontierInquiry] = None
    relevant_requirement_ids: List[str] = Field(default_factory=list)
    # Internal Python-owned carryover; the model-provided value is discarded.
    deferred_frontiers: List[dict] = Field(default_factory=list)
    feedback: Optional[ThreadFeedback] = None
    rationale: str = ""

    @field_validator("thread_id", "parent_thread_id", mode="before")
    @classmethod
    def normalize_thread_ids(cls, value):
        if value is None or not isinstance(value, str):
            return value
        normalized = re.sub(r"[^a-z0-9_.-]+", "_", value.strip().lower()).strip("_.-")
        return normalized[:80] or None


_thread_planner = None
_thread_planner_repair = None
_inquiry_assessment_model = None


class InquiryAssessment(BaseModel):
    """Semantic evidence audit for one proposed discovery frontier.

    The model reports evidence and missing information separately. Coverage is
    derived in code so "related context exists" cannot be mistaken for
    "the requested information is already known".
    """

    supporting_observation_ids: List[str] = Field(default_factory=list, max_length=12)
    recent_answer_supports: bool = False
    information_need_resolved: bool
    missing_information: List[str] = Field(default_factory=list, max_length=8)
    too_broad: bool
    recap_of_known_information: bool = False
    should_move_on: bool = False
    repeats_rejected_frontier: bool = False
    repeats_prior_decision: bool
    matching_prior_question: str = ""
    abstraction_level: Literal[
        "PRODUCT_DECISION",
        "PRODUCT_BEHAVIOR",
        "INTERACTION_DESIGN",
        "IMPLEMENTATION",
    ]
    material_product_consequence: bool
    current_frontier_value: float = Field(default=0.5, ge=0, le=1)
    best_alternative_value: float = Field(default=0.0, ge=0, le=1)
    higher_value_elsewhere: bool = False
    best_alternative_focus: str = ""
    depth_reason: str = ""
    reason: str

    @field_validator("supporting_observation_ids", "missing_information")
    @classmethod
    def unique_list_values(cls, values):
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def coherent_coverage_verdict(self):
        if self.information_need_resolved and self.missing_information:
            raise ValueError(
                "A resolved information need cannot also contain missing_information"
            )
        if not self.information_need_resolved and not self.missing_information:
            raise ValueError(
                "An unresolved information need must name the material missing information"
            )
        if self.recap_of_known_information and not self.information_need_resolved:
            raise ValueError(
                "A recap of known information must have information_need_resolved=true"
            )
        if self.higher_value_elsewhere and not self.best_alternative_focus.strip():
            raise ValueError(
                "higher_value_elsewhere requires a specific grounded alternative"
            )
        if self.repeats_prior_decision and not self.matching_prior_question.strip():
            raise ValueError(
                "repeats_prior_decision requires the matching prior delivered question"
            )
        if (
            self.abstraction_level in {"INTERACTION_DESIGN", "IMPLEMENTATION"}
            and not self.material_product_consequence
            and not self.should_move_on
        ):
            raise ValueError(
                "Low-level interaction/implementation detail without a material "
                "product consequence must set should_move_on=true"
            )
        return self


def inquiry_assessment_model():
    global _inquiry_assessment_model
    if _inquiry_assessment_model is None:
        _inquiry_assessment_model = get_structured_model(
            call_name="discovery_threads.assess_inquiry",
            schema=InquiryAssessment,
        )
    return _inquiry_assessment_model


def thread_planner_model():
    global _thread_planner
    if _thread_planner is None:
        _thread_planner = get_structured_model(
            call_name="discovery_threads.plan",
            schema=DiscoveryThreadPlan,
        )
    return _thread_planner


def thread_planner_repair_model():
    global _thread_planner_repair
    if _thread_planner_repair is None:
        _thread_planner_repair = get_structured_model(
            call_name="discovery_threads.plan_repair",
            schema=DiscoveryThreadPlan,
        )
    return _thread_planner_repair


THREAD_PLANNER_INSTRUCTION = """You choose the NEXT move in a grounded
product-manager interview. Confirmed founder facts are authoritative. Never invent
an answer, role, workflow, feature, rule, or dependency.

GOAL
Understand the founder's intended product well enough to write a useful PRD.
"An engineer could build something" is NOT a stopping rule. A simple product can
still have important product decisions.

HOW TO CHOOSE THE NEXT QUESTION
1. Read what changed in the founder's latest answer.
2. Scan pending_frontiers as well as the confirmed product model. A pending
   frontier is a previously validated material decision deferred only because
   another decision ranked higher. Do not silently forget it; choose it when it
   becomes competitive, and keep it pending until answered or explicitly deferred.
3. Prefer a causal consequence of something already established when it changes
   the product contract. Examples of material product-contract areas include:
   - user outcome / job-to-be-done. An enabling quality such as availability,
     persistence, authentication, speed, or platform access is NOT by itself a
     substitute for knowing what the user is fundamentally trying to accomplish;
   - core entities and meaningful data shape;
   - ownership, persistence, and access;
   - lifecycle states and allowed transitions;
   - validations and constraints;
   - user-visible behavior after important actions;
   - search/filter/sort when records must be found or managed;
   - reminders/notifications when time or deadlines exist;
   - platform/account boundaries;
   - permissions, business rules, money, compliance, dependencies, and important
     exception behavior.
4. Ask EXACTLY ONE independently answerable decision at a time. Do not ask for an
   end-to-end flow, multiple actors' journeys, or several unrelated choices in one
   question.
5. Stay on the active thread while its next causal decision is still among the
   highest-value unresolved decisions. Otherwise switch threads.

PRODUCT-CONTRACT MATERIALITY
A question is worth asking when two plausible founder answers would lead to
meaningfully different PRD requirements, entity/data shape, lifecycle,
validation, access/persistence, notification behavior, platform scope, or other
user-visible product behavior. Risk, money, compliance, and irreversibility are
strong signals but are NOT required.

CRUD actions are capability-level facts, not automatic closure. A follow-up such
as whether completed records remain editable or whether deletion is recoverable
can be a real product decision because it changes lifecycle/retention behavior.
Do not mechanically interrogate every CRUD possibility.

ABSTRACTION BOUNDARY
Ask product-owner questions about WHAT should happen, WHEN, WHO may act, or WHAT
rule applies. Do not ask about button placement, modal-vs-toast, visual styling,
copy, component choice, database representation, API design, SDKs, algorithms, or
other implementation mechanics unless the founder explicitly makes them a
product requirement.

FOUNDER GUIDANCE AND BOUNDARIES
founder_gap_guidance contains founder-declared unresolved areas. It is guidance,
not product knowledge. Do not silently discard an unresolved founder-named
product decision merely because the product is simple.

Respect discovery_boundaries:
- decision_deferral: move away until explicitly reopened;
- implementation/design deferral: do not ask that mechanism again;
- rejected inquiry: do not paraphrase/retry the same decision;
- question_too_broad: preserve the area but ask one smaller decision;
- product_scope_closed: stop discovering that line unless reopened;
- generation_exhausted: choose a materially different grounded inquiry.

If the latest founder answer is interview feedback rather than product content,
return feedback describing that boundary without inventing a product fact.

REQUIREMENTS
eligible_requirement_backlog is a supplied backlog, not a checklist. Include only
real supplied requirement IDs that are relevant NOW. Never invent IDs.

STOPPING TEST
Apply this only AFTER scanning the product model. Return frontier=null only when
the PRD can describe the founder's intended product without the team having to
choose among materially different user-facing/product-model interpretations, and
the remaining unknowns are principally design, implementation, or low-impact
preference. Do not treat an access/persistence/platform quality as sufficient
evidence of the user's core job-to-be-done; if the product's fundamental user
outcome is still unknown, that remains a material discovery frontier.

OUTPUT
Choose stable semantic thread_id and decision_key values. The frontier is an
uncertainty, never an invented answer. anchor_gap is optional metadata; use null
when no legacy field fits. Keep objective/question_hint founder-facing and
product-level.
"""


def _latest_human(state: AgentState) -> str:
    return next(
        (
            message_text(message.content)
            for message in reversed(state.get("messages", []))
            if isinstance(message, HumanMessage)
        ),
        state.get("raw_idea", ""),
    )


def _fact_payload(state: AgentState, scope: DiscoveryScope) -> list[dict]:
    result = []
    for item in state.get("discovered_knowledge", []):
        if (
            item.scope != scope
            or item.knowledge_state != KnowledgeState.CONFIRMED
        ):
            continue
        result.append({
            "fact_id": fact_id(item),
            "topic": item.topic.value,
            "key": item.key,
            "role": item.role,
            "roles": item.roles,
            "aliases": item.aliases,
            "value": item.value,
            "source_turn": item.source_turn,
        })
    return result[-40:]


def _latest_turn_facts(state: AgentState, scope: DiscoveryScope) -> list[dict]:
    turn = state.get("turn_count", 0)
    return [
        item
        for item in _fact_payload(state, scope)
        if item.get("source_turn") == turn
    ]


def _latest_turn_concepts(state: AgentState, scope: DiscoveryScope) -> list[dict]:
    turn = state.get("turn_count", 0)
    return [
        item
        for item in _concept_payload(state, scope)
        if item.get("source_turn") == turn
    ]


def _concept_payload(state: AgentState, scope: DiscoveryScope) -> list[dict]:
    result = []
    for raw in state.get("product_concepts", []):
        concept = raw if isinstance(raw, ProductConcept) else ProductConcept.model_validate(raw)
        if concept.scope != scope:
            continue
        result.append(concept.model_dump(mode="json"))
    return result[-40:]


def _external_system_payload(state: AgentState, scope: DiscoveryScope) -> list[dict]:
    result = []
    for raw in state.get("external_systems", []) or []:
        system = raw if isinstance(raw, ExternalSystem) else ExternalSystem.model_validate(raw)
        if system.scope != scope:
            continue
        result.append(system.model_dump(mode="json"))
    return result[-30:]


def _latest_turn_external_systems(
    state: AgentState,
    scope: DiscoveryScope,
) -> list[dict]:
    turn = state.get("turn_count", 0)
    result = []
    for system in _external_system_payload(state, scope):
        latest_statements = [
            statement
            for statement in system.get("statements", [])
            if statement.get("source_turn") == turn
        ]
        if latest_statements:
            result.append({**system, "statements": latest_statements})
    return result


def _observation_payload(state: AgentState, scope: DiscoveryScope) -> list[dict]:
    """Founder evidence survives even when canonical schema admission rejects it."""
    result = []
    for observation in state.get("captured_observations", []):
        if observation.get("scope") != scope.value:
            continue
        result.append({
            "id": observation.get("id"),
            "kind": observation.get("kind"),
            "value": observation.get("value"),
            "evidence": observation.get("evidence"),
            "role": observation.get("role"),
            "source_turn": observation.get("source_turn"),
            "admission_status": observation.get("admission_status"),
        })
    return result[-30:]


def _requirement_payload(state: AgentState, scope: DiscoveryScope) -> list[dict]:
    eligible = set(state.get("eligible_requirement_keys", []))
    result = []
    for key, requirement in state.get("active_requirements", {}).items():
        if requirement.scope != scope or key not in eligible:
            continue
        coverage = state.get("requirement_coverage", {}).get(key, {})
        unresolved = [
            facet.id
            for facet in requirement.facets
            if facet.required
            and (coverage.get("facets", {}).get(facet.id, {}).get("state") == "UNKNOWN"
                 or facet.id not in coverage.get("facets", {}))
        ]
        dependency = state.get("requirement_dependency_state", {}).get(key, {})
        result.append({
            "requirement_key": key,
            "requirement_id": requirement.id,
            "label": requirement.label,
            "description": requirement.description,
            "topic": requirement.topic.value,
            "parent_gap": requirement.parent_gap,
            "unresolved_facets": unresolved,
            "dependencies": list(requirement.dependencies),
            "unlocks": list(requirement.unlocks),
            "dependency_eligible": dependency.get("eligible"),
            "architecture_impact": requirement.priority_hints.architecture_impact,
            "business_risk": requirement.priority_hints.business_risk,
        })
    return result[:30]


def _recent_conversation(state: AgentState) -> list[dict]:
    # The planner already receives durable product knowledge. Only a short local
    # conversational window is needed for continuity and pronoun resolution.
    result = []
    for message in state.get("messages", [])[-4:]:
        if isinstance(message, HumanMessage):
            result.append({"speaker": "founder", "text": message_text(message.content)})
        elif isinstance(message, AIMessage):
            result.append({"speaker": "pm", "text": message_text(message.content)})
    return result


def _history_payload(state: AgentState) -> list[dict]:
    result = []
    for entry in state.get("requirement_question_history", [])[-12:]:
        result.append({
            "turn": entry.get("turn"),
            "thread_id": entry.get("thread_id"),
            "decision_key": entry.get("decision_key"),
            "inquiry_id": entry.get("inquiry_id"),
            "objective": entry.get("objective"),
            "question": entry.get("question"),
        })
    return result


def _gap_guidance_payload(
    state: AgentState,
    scope: DiscoveryScope,
) -> list[dict]:
    result = []
    for entry in state.get("founder_gap_guidance", []) or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("scope") and entry.get("scope") != scope.value:
            continue
        result.append({
            "scope": entry.get("scope") or scope.value,
            "source_turn": entry.get("source_turn"),
            "evidence": entry.get("evidence"),
            "items": entry.get("items", []),
            "instruction": entry.get("instruction"),
        })
    return result[-8:]


def _compact_product_snapshot(
    state: AgentState,
    scope: DiscoveryScope,
) -> dict:
    """Planner-facing semantic state without provenance/evidence duplication.

    Full evidence remains in graph state and is used by extraction/validation.
    The expensive reasoning model only needs the current product contract.
    """
    grouped_facts: dict[str, list[str]] = {}
    for item in state.get("discovered_knowledge", []):
        if (
            item.scope != scope
            or item.knowledge_state != KnowledgeState.CONFIRMED
        ):
            continue
        actor = item.role or ",".join(item.roles or [])
        key = f"{item.topic.value}.{item.key}" + (f"[{actor}]" if actor else "")
        values = grouped_facts.setdefault(key, [])
        value = str(item.value).strip()
        if item.absence:
            value = f"{value} [explicit absence: {item.absence}]"
        if value and value not in values:
            values.append(value)
            if len(values) > 8:
                del values[0]

    compact_concepts: list[dict] = []
    seen_concepts: set[tuple] = set()
    for raw in state.get("product_concepts", []) or []:
        try:
            concept = raw if isinstance(raw, ProductConcept) else ProductConcept.model_validate(raw)
        except Exception:
            continue
        if concept.scope != scope:
            continue
        item = {
            "kind": concept.kind.value,
            "subject": concept.subject,
            "relation": concept.relation,
            "object": concept.object,
            "value": concept.value,
        }
        signature = tuple(item.values())
        if signature in seen_concepts:
            continue
        seen_concepts.add(signature)
        compact_concepts.append(item)
    compact_concepts = compact_concepts[-30:]

    systems = []
    for raw in state.get("external_systems", []) or []:
        try:
            system = raw if isinstance(raw, ExternalSystem) else ExternalSystem.model_validate(raw)
        except Exception:
            continue
        if system.scope != scope:
            continue
        systems.append({
            "name": system.name,
            "statements": [
                statement.value
                for statement in system.statements[-6:]
                if statement.value.strip()
            ],
        })

    return {
        "facts_by_category": grouped_facts,
        "product_concepts": compact_concepts,
        "external_systems": systems[-10:],
    }


def _compact_threads(state: AgentState) -> list[dict]:
    raw_threads = state.get("discovery_threads", {}) or {}
    active = state.get("active_discovery_thread")
    items: list[dict] = []
    for thread_id, raw in raw_threads.items():
        data = (
            raw.model_dump(mode="json")
            if hasattr(raw, "model_dump")
            else dict(raw)
            if isinstance(raw, dict)
            else {}
        )
        if not data:
            continue
        items.append({
            "id": thread_id,
            "label": data.get("label"),
            "objective": data.get("objective"),
            "status": getattr(data.get("status"), "value", data.get("status")),
            "last_active_turn": data.get("last_active_turn", 0),
            "active": thread_id == active,
        })
    items.sort(
        key=lambda item: (not item["active"], -(item["last_active_turn"] or 0))
    )
    return items[:6]


def _planner_observations(
    state: AgentState,
    scope: DiscoveryScope,
) -> list[dict]:
    """Keep only recent observations that add evidence beyond canonical state."""
    observations = _observation_payload(state, scope)
    useful = [
        item
        for item in observations
        if item.get("admission_status") not in {"ADMITTED", "CONCEPT"}
    ]
    # Also retain a few latest observations for continuity/debugging.
    latest = observations[-6:]
    merged: list[dict] = []
    seen = set()
    for item in [*useful[-12:], *latest]:
        identity = item.get("id")
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(item)
    return merged[-16:]


def _thread_activity_payload(state: AgentState) -> dict:
    """Expose conversation depth as evidence, never as a hard stopping rule."""
    history = state.get("requirement_question_history", [])
    active = state.get("active_discovery_thread")
    counts: dict[str, int] = {}
    for entry in history:
        thread_id = entry.get("thread_id")
        if thread_id:
            counts[thread_id] = counts.get(thread_id, 0) + 1

    streak = 0
    if active:
        for entry in reversed(history):
            if entry.get("thread_id") != active:
                break
            streak += 1

    return {
        "active_thread_id": active,
        "questions_by_thread": counts,
        "consecutive_questions_on_active_thread": streak,
        "note": (
            "Counts are context only. Do not move on because a numeric limit was reached; "
            "use them to notice sustained drilling and compare marginal value."
        ),
    }


def _normalize_plan(
    plan: DiscoveryThreadPlan,
    state: AgentState,
    scope: DiscoveryScope,
) -> DiscoveryThreadPlan:
    known_fact_ids = {item["fact_id"] for item in _fact_payload(state, scope)}
    known_requirement_ids = {
        item["requirement_id"] for item in _requirement_payload(state, scope)
    }
    relevant = [
        requirement_id
        for requirement_id in plan.relevant_requirement_ids
        if requirement_id in known_requirement_ids
    ]
    frontier = plan.frontier
    if frontier is not None:
        frontier = frontier.model_copy(update={
            "related_fact_ids": [
                identity for identity in frontier.related_fact_ids
                if identity in known_fact_ids
            ],
        })
    parent = plan.parent_thread_id
    if parent == plan.thread_id:
        parent = None
    return plan.model_copy(update={
        "parent_thread_id": parent,
        "frontier": frontier,
        "relevant_requirement_ids": relevant,
        "deferred_frontiers": [],
    })


INQUIRY_ASSESSMENT_INSTRUCTION = """Assess the proposed next discovery
frontier before a question is generated.

Do NOT decide coverage from topical similarity. Separate what is KNOWN from what
the proposed frontier still asks the founder to supply. A confirmed answer resolves
a frontier only when it matches the same triggering event and preconditions.
Specifically distinguish setting a due date, editing a due date, completing a task,
reopening a completed task, and a deadline passing. A rule for reminders after
reopening a task does NOT answer what happens when its due date is edited or passes.

founder_gap_guidance, when supplied, is meta-level founder guidance about
questions/areas that remain open. It is not evidence that any answer is true.
Use it only to recognize founder-prioritized unresolved areas. Coverage must still
come from confirmed facts/observations or the latest direct answer.

Return:
- supporting_observation_ids: only UNIQUE observation IDs whose founder evidence
  DIRECTLY answers some or all of the proposed information need. Never repeat an
  ID and return at most 12 IDs.
- recent_answer_supports=true when the immediately preceding founder answer
  directly resolves the proposed information need through conversational context,
  even if that short answer is not a stored observation.
- information_need_resolved=true ONLY when the ENTIRE proposed information need
  is directly answered by founder evidence. If any material part remains unknown,
  it MUST be false.
- missing_information: every material part of the proposed information need that
  is NOT directly answered by existing founder evidence, stated once each (at most
  8 items). If your reason says
  something is unknown, unspecified, not explained, or still needed, that item
  MUST appear here. Never return an empty list while describing missing detail.
- recap_of_known_information=true ONLY when the proposed frontier asks the
  founder to restate/summarize information that is already directly present and
  there is no material new information to obtain.
- too_broad=true when the frontier contains more than ONE independently
  answerable uncertainty. This includes timing + process + conditions;
  permissions + features + experience; multiple workflow stages; multiple actors'
  journeys; or independent responsibilities/permissions/goals. It ALSO includes
  a request for "main actions", "main steps", "the process", "the flow", or a
  start-to-end workflow when answering naturally requires a sequence of several
  actions. A single grammatical question can still be too broad. If the founder
  could answer one requested part while leaving another unanswered, it is too
  broad. Choose one atomic fork, state, relationship, rule, or causal link.
- current_frontier_value: assess the marginal value of asking THIS frontier now,
  considering information gain, product/business/architecture/risk impact,
  dependency unlock value, and question cost.
- best_alternative_value: assess the strongest materially unresolved question
  available OUTSIDE the proposed thread using the supplied eligible requirement
  backlog, paused/current threads, and confirmed product model. Use 0 only when
  there is no grounded alternative.
- higher_value_elsewhere=true when a specific grounded alternative appears more
  valuable than this frontier now. Name that area/decision in
  best_alternative_focus. This is DIAGNOSTIC ONLY: the planner already owns global
  breadth-vs-depth ranking, so do not invalidate an otherwise good frontier merely
  because you can imagine a somewhat better alternative.
- low_signal_crud_hint is a NON-AUTHORITATIVE heuristic indicating that the
  question resembles CRUD refinement in a simple context. Consider it, but do
  not treat it as proof of low value. The product-contract delta test is
  authoritative.
- should_move_on=true ONLY when the proposed frontier itself is no longer worth
  asking at this discovery stage: for example it is implementation/UI detail,
  exhaustive refinement after the governing rule is already coherent, or continued
  drilling whose answer would not materially change the product model. Do NOT set
  should_move_on merely because another valid frontier might rank higher.
- repeats_rejected_frontier=true when the proposed frontier is semantically the
  same underlying decision as any item in rejected_frontiers, even if its
  decision_key or wording changed. A rejected/covered frontier is closed for this
  repair attempt and must not be selected again.
- repeats_prior_decision=true when the proposed frontier is semantically the same
  governing uncertainty as ANY delivered question in delivered_question_history,
  even when thread_id, decision_key, wording, or schema anchor differ. This is
  about meaning, not string similarity. Set matching_prior_question to the most
  relevant delivered question. Do NOT mark a genuine next causal decision as a
  repeat merely because it occurs in the same workflow.
- abstraction_level:
  PRODUCT_DECISION = product-shape choice/rule/actor/goal/constraint;
  PRODUCT_BEHAVIOR = material state/outcome/validation/authorization/dependency;
  INTERACTION_DESIGN = screen flow, button/control sequence, layout, placement,
  presentation, microcopy, clickable-vs-text, modal/toast/component choice;
  IMPLEMENTATION = technical mechanism/architecture/code/service internals.
- material_product_consequence=true when knowing this answer could materially
  change ANY grounded product-contract element: a functional requirement,
  user-visible capability/availability, entity/data shape, ownership/persistence,
  validation/constraint, lifecycle/state transition, notification/reminder rule,
  platform/access boundary, business rule, authorization/security/compliance
  boundary, money/data movement, major dependency, or similarly meaningful PRD
  behavior. The product does NOT need to be high-risk, multi-user, financial, or
  irreversible for a consequence to be material. If abstraction_level is
  INTERACTION_DESIGN or IMPLEMENTATION and the answer would not change the
  product contract, set should_move_on=true even when the detail is unknown.

CRITICAL COVERAGE RULE:
Related context is NOT an answer. Knowing WHO the actors are does not answer WHAT
their responsibilities, permissions, goals, or workflows are. Knowing the product
category does not answer who its actors are or what its workflow is. Knowing one
stage of a process does not answer a different stage.

If you can truthfully say "the founder has not provided X yet", then X MUST appear
in missing_information and recap_of_known_information MUST be false.

Interpret concise answers against the immediately preceding PM question. If the PM
asked "are there any others?" and the founder replies "that's all", "that will be
all", "nothing else", or equivalent, that IS an explicit closure of that list:
recent_answer_supports=true and do not invent a need for another confirmation.

End-to-end requests such as "walk me through the main steps from X to Y" or
"what are the main actions A and B perform from X to Y?" are too broad when X->Y
spans several actions/decisions. The latter is too broad even though it appears
as one sentence, because it combines multiple actors and multiple stages.
Likewise, "when and how",
"what changes in X, Y, and Z", or "what process applies and what conditions are
required" are bundled inquiries unless they describe one indivisible choice.
A valid frontier should require ONE substantive answer, not a checklist.

When an existing canonical role's capabilities are already known and the founder
says another user becomes that role, do not ask them to restate that role's
features or permissions unless the evidence suggests the transitioned user is
different. If that distinction is genuinely uncertain, ask only whether they
become the same role or a distinct/limited version of it.

DEPTH / MARGINAL VALUE:
Product discovery is not an exhaustive interrogation. "Something is still
unknown" is NOT enough reason to keep asking inside the same thread.

Do not equate "ordinary CRUD" with "fully specified product behavior." The action
itself may be known while an adjacent product decision remains material. Ask the
follow-up when plausible answers would create different functional requirements,
entity/data shape, lifecycle/state rules, validations, access/persistence rules,
notifications, platform scope, or other user-visible product behavior.

For simple or single-actor products, judge marginal value exactly the same way:
by PRODUCT-CONTRACT DELTA, not by risk. Simplicity is not a reason to stop early.
A low-risk decision may still define the product. Conversely, details that only
change layout, wording, component choice, exact navigation choreography, or
internal implementation remain low-value unless founder evidence makes them part
of the product contract.

A thread is coherent enough to pause when its governing product shape can be
represented without guessing: the important actors/entities, the core relation
or rule, and the material state/decision currently being discussed are clear
enough that remaining questions mostly refine rather than reshape it. This is a
semantic judgment, not a required-field checklist.

The PLANNER, not this assessor, performs the global comparison against unresolved
decisions elsewhere. You may report a stronger alternative as diagnostic context,
but this assessor must not turn ranking disagreement into invalidity. Its job is
to reject a frontier only when the frontier itself is covered, repeated, too
broad, boundary-violating, or intrinsically too low-value/deep for the current
discovery stage.

Do not use a fixed question-count cutoff. Thread/question counts are evidence of
possible drilling, not a stopping rule. Once the governing product rule is clear,
do not keep drilling merely because more implementation detail, examples, UI
mechanics, exhaustive enumeration, or edge specificity could exist. "Make the
question smaller" must NOT become "make it more screen-specific"; decomposition
stops when the remaining uncertainty is principally interaction design rather
than a material product decision. If recent
questions have stayed on the same narrow area and the next answer would not
materially change the product model, PRD decision, business rule, architecture
boundary, money movement, authorization model, lifecycle, or major dependency,
prefer a higher-value unresolved area. A founder request to avoid technical depth
is also strong evidence that implementation-level follow-ups should stop.

Only founder statements count as evidence. PM questions do not. Captured observations remain useful when canonical STRUCTURAL admission
failed, because their exact founder evidence still exists. However, an observation
with admission_status="SEMANTIC_REJECTED" means the LLM grounding audit found that
the proposed interpretation was not supported. Do NOT use that observation's
kind/value as direct support for an inquiry. Its raw evidence may only be treated
as founder text and must be interpreted independently.

Do not invent missing information and do not treat implications from a product
label as founder-provided facts.

Discovery boundaries are conversation-control constraints, not missing product
facts. If a proposed frontier asks for detail the founder explicitly delegated
to a designer/implementation specialist, repeats/deepens an inquiry they
rejected as irrelevant, repeats an active decision_deferral, or enters an area
marked product_scope_closed, set should_move_on=true even if that detail remains
unknown. A decision_deferral is intentionally unresolved and must not be treated
as ordinary missing information until its boundary is explicitly reopened.
A question_too_broad
boundary does not close the underlying product area; it only forbids asking for
the same oversized bundle again. A product_scope_closed boundary DOES close
that line of discovery until later founder evidence explicitly reopens it.
Unknown does not mean worth asking.

"""


CRUD_ACTION_PATTERN = re.compile(
    r"\b(?:create|add|remove|delete|edit|update|rename|mark|complete|view|"
    r"check|uncheck|archive|restore|reopen)\w*\b",
    re.I,
)
CRUD_DEPTH_PATTERN = re.compile(
    r"\b(?:rules?|restrictions?|conditions?|constraints?|limits?|validations?|"
    r"fields?|information|attributes?|aspects?|parts?|anything\s+else|"
    r"special\s+behavio[u]?rs?|undo|restore|restoration|recover|recovery|"
    r"permanent(?:ly)?|grace\s+period|history|historical|retention|retain|"
    r"archive|archived|lock|locked|confirmation|confirm|reopen|"
    r"revers(?:e|ible|ibility)|side\s+effects?|additional\s+behavio[u]?r|"
    r"what\s+happens(?:\s+next|\s+immediately)?|who\s+(?:can|should)|"
    r"allowed|authoriz(?:e|ed|ation)|display|interactions?|after\s+completion|"
    r"completed\s+(?:state|list|item|task)|timing|immediately|invalid|error|"
    r"message|toast|notification|feedback|guidance|automatic|automatically|"
    r"reset|clear|focus|screen|wording|look\s+like|cancel|abort|"
    r"during\s+creation|at\s+creation)\b",
    re.I,
)
LOW_RISK_INTERACTION_DETAIL_PATTERN = re.compile(
    r"\b(?:confirmation\s+message|success\s+message|toast|notification|"
    r"microcopy|wording|look\s+like|how\s+should\s+the\s+user\s+interact|"
    r"input\s+form|reset|clear(?:ing)?|focus|screen\s+changes?|ui\s+updates?|"
    r"automatic\s+(?:action|update)|immediate\s+ui|post[- ](?:creation|"
    r"deletion|completion)|error\s+message|user\s+guidance)\b",
    re.I,
)
CREATION_SHAPE_PATTERN = re.compile(
    r"\b(?:create|creation|add)\w*\b.*\b(?:field|information|attribute|"
    r"provide|enter|required|validation)\w*\b",
    re.I,
)
ENTRY_POINT_PATTERN = re.compile(
    r"\b(?:first\s+(?:meaningful\s+)?action|initial\s+action|entry\s+point|"
    r"how\s+(?:does|should).{0,30}\b(?:start|begin)|what.{0,30}\b(?:start|begin)"
    r"|trigger.{0,20}(?:workflow|interaction))\b",
    re.I,
)

MATERIAL_COMPLEXITY_PATTERN = re.compile(
    r"\b(?:payment|pay|paid|money|fund|escrow|approval|approve|authoriz|"
    r"permission|compliance|legal|regulat|security|identity|verification|"
    r"dispute|contract|ownership|entitlement|audit|retention|retain|"
    r"external|integration|inventory|capacity)\w*\b",
    re.I,
)


def _positive_materiality_fact(item: KnowledgeItem) -> bool:
    """Only positive in-scope facts may make discovery materially complex."""
    if item.absence:
        return False
    if item.topic == DiscoveryTopic.MVP_SCOPE and item.key == "out_of_scope":
        return False
    return bool(
        MATERIAL_COMPLEXITY_PATTERN.search(f"{item.value} {item.evidence}")
    )


def _low_risk_single_actor_context(
    state: AgentState,
    scope: DiscoveryScope,
) -> bool:
    knowledge = [
        item
        for item in state.get("discovered_knowledge", [])
        if item.scope == scope
        and item.knowledge_state == KnowledgeState.CONFIRMED
    ]
    primary_roles = {
        role
        for item in knowledge
        if item.topic == DiscoveryTopic.USER_ROLES
        and item.key == "primary_users"
        and not item.absence
        for role in (item.roles or [])
    }
    secondary_positive = any(
        item.topic == DiscoveryTopic.USER_ROLES
        and item.key == "secondary_users"
        and not item.absence
        for item in knowledge
    )
    secondary_absent = any(
        item.topic == DiscoveryTopic.USER_ROLES
        and item.key == "secondary_users"
        and item.absence
        for item in knowledge
    )
    role_complexity = any(
        item.topic == DiscoveryTopic.USER_ROLES
        and item.key in {"multiple_roles", "role_transitions"}
        and not item.absence
        for item in knowledge
    )
    material_fact = any(_positive_materiality_fact(item) for item in knowledge)
    external_systems = []
    for item in state.get("external_systems", []) or []:
        raw_scope = (
            item.get("scope")
            if isinstance(item, dict)
            else getattr(item, "scope", None)
        )
        normalized_scope = getattr(raw_scope, "value", raw_scope)
        if normalized_scope == scope.value:
            external_systems.append(item)
    return bool(
        len(primary_roles) == 1
        and secondary_absent
        and not secondary_positive
        and not role_complexity
        and not material_fact
        and not external_systems
    )


def _creation_shape_already_known(
    state: AgentState,
    scope: DiscoveryScope,
) -> bool:
    for item in state.get("discovered_knowledge", []):
        if (
            item.scope != scope
            or item.knowledge_state != KnowledgeState.CONFIRMED
            or item.absence
        ):
            continue
        text = f"{item.value} {item.evidence}".lower()
        if (
            item.key == "validation_rules"
            or (
                item.key in {"responsibilities", "workflow_steps", "must_have_features"}
                and re.search(
                    r"\b(?:provide|enter|required|field|title|name|attribute)\w*\b",
                    text,
                )
            )
        ):
            return True
    return False


CRUD_ACTION_FAMILIES = {
    "create": ("create", "add", "make"),
    "remove": ("remove", "delete"),
    "edit": ("edit", "update", "rename", "modify", "change"),
    "complete": ("mark", "complete", "check", "uncheck"),
    "view": ("view", "see", "display", "show"),
    "archive": ("archive",),
    "restore": ("restore", "reopen"),
}


def _crud_action_families(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", (text or "").lower())
    families: set[str] = set()
    for index, word in enumerate(words):
        # "completed" frequently describes the state of a task/item rather than
        # an action: "edit completed tasks", "delete completed items". Do not
        # let that adjective manufacture a COMPLETE action family.
        if (
            word == "completed"
            and index + 1 < len(words)
            and words[index + 1] in {"task", "tasks", "item", "items", "list", "lists"}
        ):
            continue
        for family, prefixes in CRUD_ACTION_FAMILIES.items():
            if any(word.startswith(prefix) for prefix in prefixes):
                families.add(family)
    return families


def _known_ordinary_crud_action(
    proposal: str,
    state: AgentState,
    scope: DiscoveryScope,
) -> bool:
    """Whether this frontier is drilling into an already-confirmed CRUD action.

    This intentionally does not require actor discovery to be complete. Once the
    founder has explicitly established an ordinary action such as remove/delete,
    a checklist about its default restrictions/conditions is low marginal value
    unless the product context already contains material complexity.
    """
    proposal_actions = _crud_action_families(proposal)
    if not proposal_actions:
        return False

    knowledge = [
        item
        for item in state.get("discovered_knowledge", [])
        if item.scope == scope
        and item.knowledge_state == KnowledgeState.CONFIRMED
        and not item.absence
    ]
    if any(_positive_materiality_fact(item) for item in knowledge):
        return False

    if any(
        getattr(
            item.get("scope") if isinstance(item, dict) else getattr(item, "scope", None),
            "value",
            item.get("scope") if isinstance(item, dict) else getattr(item, "scope", None),
        ) == scope.value
        for item in state.get("external_systems", []) or []
    ):
        return False

    return any(
        item.key in {"responsibilities", "workflow_steps", "must_have_features"}
        and bool(
            proposal_actions
            & _crud_action_families(f"{item.value} {item.evidence}")
        )
        for item in knowledge
    )


def _low_signal_crud_depth_frontier(
    plan: DiscoveryThreadPlan,
    state: AgentState,
    scope: DiscoveryScope,
) -> bool:
    """Reject founder interrogation about ordinary CRUD defaults in simple products."""
    frontier = plan.frontier
    if frontier is None:
        return False
    proposal = " ".join(
        filter(None, [frontier.objective, frontier.question_hint, frontier.reason])
    )
    low_risk = _low_risk_single_actor_context(state, scope)

    # For a genuinely low-risk product, micro-interaction choices belong to
    # design/engineering unless the founder volunteered a material consequence.
    # They must never be promoted into founder interview obligations merely
    # because the planner can imagine another UI decision.
    if low_risk and LOW_RISK_INTERACTION_DETAIL_PATTERN.search(proposal):
        return True

    # Asking for the start/entry point is a recap when a low-risk product already
    # has an explicit create/add responsibility. The planner should lift to a
    # different product-shape uncertainty or stop instead.
    if low_risk and ENTRY_POINT_PATTERN.search(proposal):
        has_known_entry_action = any(
            item.scope == scope
            and item.knowledge_state == KnowledgeState.CONFIRMED
            and not item.absence
            and item.key in {"responsibilities", "workflow_steps"}
            and re.search(r"\b(?:create|add|submit|start|begin)\w*\b", item.value, re.I)
            for item in state.get("discovered_knowledge", [])
        )
        if has_known_entry_action:
            return True

    if not (
        CRUD_ACTION_PATTERN.search(proposal)
        and CRUD_DEPTH_PATTERN.search(proposal)
    ):
        return False

    # One creation/entity-shape question can be material: the team may genuinely
    # need to know what object the founder is asking them to create. Once creation
    # shape/validation is known, do not mirror the same checklist across edit,
    # delete, completion, history, and display behavior.
    if (
        CREATION_SHAPE_PATTERN.search(proposal)
        and not _creation_shape_already_known(state, scope)
    ):
        return False

    return bool(
        _low_risk_single_actor_context(state, scope)
        or _known_ordinary_crud_action(proposal, state, scope)
    )



def _decision_trigger_family(text: str | None) -> str | None:
    """Classify explicit lifecycle triggers, not shared nouns like 'task' or 'reminder'."""
    normalized = re.sub(r"\s+", " ", (text or "").lower())
    if (
        re.search(r"\b(?:reopen(?:ed|ing)?|returned to active|changed back to active|from completed to active)\b", normalized)
        and re.search(r"\bcompleted\b", normalized)
    ):
        return "reopen_completed_task"
    if (
        re.search(r"\b(?:due date|due time|deadline)\b", normalized)
        and re.search(r"\b(?:passes?|passed|reaches?|reached|after|elapsed|overdue|without being completed)\b", normalized)
    ):
        return "deadline_passed"
    if (
        re.search(r"\b(?:delete|deletion|remove|removed)\b", normalized)
        and re.search(r"\btask\b", normalized)
    ):
        return "delete_task"
    if (
        re.search(r"\b(?:edit|edited|editing|modify|modified|update|updated|change|changed)\b", normalized)
        and re.search(r"\b(?:due date|due time|deadline)\b", normalized)
    ):
        return "edit_due_date"
    if (
        re.search(r"\b(?:set|sets|setting|assign|assigned)\b", normalized)
        and re.search(r"\b(?:due date|due time|deadline)\b", normalized)
    ):
        return "set_due_date"
    if (
        re.search(r"\b(?:complete|completed|completion|mark(?:ed|ing)?)\b", normalized)
        and re.search(r"\btask\b", normalized)
    ):
        return "complete_task"
    return None


def _frontier_trigger_mismatch(
    plan: DiscoveryThreadPlan,
    state: AgentState,
    scope: DiscoveryScope,
) -> str | None:
    """Prevent one event's confirmed rule from closing a different event's question."""
    frontier = plan.frontier
    if frontier is None:
        return None
    target_family = _decision_trigger_family(
        " ".join([frontier.decision_key, frontier.objective, frontier.question_hint])
    )
    if target_family is None:
        return None

    confirmed_families: set[str] = set()
    for item in state.get("discovered_knowledge", []) or []:
        if (
            item.scope != scope
            or item.knowledge_state != KnowledgeState.CONFIRMED
            or item.absence
        ):
            continue
        family = _decision_trigger_family(
            " ".join([item.source_question or "", item.value or "", item.evidence or ""])
        )
        if family:
            confirmed_families.add(family)

    if target_family in confirmed_families:
        return None
    other = ", ".join(sorted(confirmed_families)) or "no explicitly captured lifecycle trigger"
    return (
        f"The proposed decision concerns '{target_family}', but confirmed facts only "
        f"identify '{other}'. Related task/reminder wording is not evidence for the same trigger."
    )


def _semantic_frontier_problem(
    plan: DiscoveryThreadPlan,
    state: AgentState,
    scope: DiscoveryScope,
    rejected_frontiers: list[dict] | None = None,
) -> str | None:
    frontier = plan.frontier
    if frontier is None:
        return None

    low_signal_crud_hint = _low_signal_crud_depth_frontier(
        plan,
        state,
        scope,
    )

    for boundary in state.get("discovery_boundaries", []) or []:
        if not isinstance(boundary, dict):
            continue
        if boundary.get("type") != "decision_deferral" or boundary.get("reopened_at_turn"):
            continue
        if boundary.get("scope") and boundary.get("scope") != scope.value:
            continue
        if (
            boundary.get("decision_key")
            and frontier.decision_key
            and boundary.get("decision_key") == frontier.decision_key
        ):
            return (
                "The proposed frontier exactly matches an explicitly deferred "
                "founder decision. Choose a materially different product decision "
                "unless the founder reopens this deferral."
            )

    trigger_alignment = _frontier_trigger_mismatch(plan, state, scope)
    payload = {
        "trigger_alignment": trigger_alignment,
        "proposed_frontier": {
            "thread_id": plan.thread_id,
            "decision_key": frontier.decision_key,
            "objective": frontier.objective,
            "question_hint": frontier.question_hint,
            "reason": frontier.reason,
        },
        "low_signal_crud_hint": low_signal_crud_hint,
        "captured_founder_observations": _observation_payload(state, scope),
        "founder_gap_guidance": _gap_guidance_payload(state, scope),
        "discovery_boundaries": [
            item for item in state.get("discovery_boundaries", [])[-50:]
            if not item.get("scope") or item.get("scope") == scope.value
        ],
        "latest_conversation_intent": state.get("conversation_intent"),
        "confirmed_product_facts": _fact_payload(state, scope),
        "confirmed_product_concepts": _concept_payload(state, scope),
        "recent_conversation": _recent_conversation(state),
        "delivered_question_history": _history_payload(state),
        "thread_activity": _thread_activity_payload(state),
        "eligible_requirement_backlog": _requirement_payload(state, scope),
        "current_threads": state.get("discovery_threads", {}),
        "rejected_frontiers": rejected_frontiers or [],
    }
    print("\n===== DISCOVERY ABSTRACTION DEBUG | STAGE 1: PLANNER FRONTIER =====")
    print(json.dumps(payload["proposed_frontier"], ensure_ascii=False, indent=2, default=str))
    print("===== END STAGE 1 =====\n")
    print("===== DISCOVERY ABSTRACTION DEBUG | STAGE 2: ASSESSOR INPUT =====")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print("===== END STAGE 2 =====\n")

    try:
        result = inquiry_assessment_model().invoke([
            SystemMessage(content=INQUIRY_ASSESSMENT_INSTRUCTION),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
        assessment = (
            result
            if isinstance(result, InquiryAssessment)
            else InquiryAssessment.model_validate(result)
        )
        print("===== DISCOVERY ABSTRACTION DEBUG | STAGE 3: ASSESSOR OUTPUT =====")
        print(json.dumps(assessment.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str))
        print("===== END STAGE 3 =====\n")
    except Exception as exc:
        raise_if_llm_failure(exc)
        print(f"INQUIRY ASSESSMENT REPAIR: inconsistent structured verdict | {exc}")
        try:
            repaired_result = inquiry_assessment_model().invoke([
                SystemMessage(content=INQUIRY_ASSESSMENT_INSTRUCTION + """
The previous assessment was structurally inconsistent. Return one corrected
assessment. If ANY material information is still unknown, set
information_need_resolved=false and list every such item in missing_information.
If the need is fully resolved, missing_information must be empty. Supporting
observation IDs must be UNIQUE and limited to the supplied IDs. Do not repeat an
ID. You MUST also return repeats_prior_decision, abstraction_level, and
material_product_consequence using the definitions in the main instruction.
Do not change the proposed frontier."""),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ])
            assessment = (
                repaired_result
                if isinstance(repaired_result, InquiryAssessment)
                else InquiryAssessment.model_validate(repaired_result)
            )
            print("===== DISCOVERY ABSTRACTION DEBUG | STAGE 3R: REPAIRED ASSESSOR OUTPUT =====")
            print(json.dumps(assessment.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str))
            print("===== END STAGE 3R =====\n")
        except Exception as repair_exc:
            raise_if_llm_failure(repair_exc)
            print(f"INQUIRY ASSESSMENT INVALID AFTER REPAIR: {repair_exc}")
            return (
                "The semantic inquiry assessment could not produce a coherent "
                "coverage verdict. Do not ask this frontier; choose a different "
                "grounded product decision."
            )

    if trigger_alignment and (
        assessment.information_need_resolved
        or assessment.recent_answer_supports
        or assessment.supporting_observation_ids
    ):
        print(f"TRIGGER ALIGNMENT OVERRIDE: {trigger_alignment}")
        assessment = assessment.model_copy(update={
            "information_need_resolved": False,
            "missing_information": [trigger_alignment],
            "recent_answer_supports": False,
            "supporting_observation_ids": [],
            "recap_of_known_information": False,
        })

    valid_observation_ids = {
        item.get("id")
        for item in payload["captured_founder_observations"]
        if item.get("id")
    }
    assessment.supporting_observation_ids = [
        identity
        for identity in dict.fromkeys(assessment.supporting_observation_ids)
        if identity in valid_observation_ids
    ][:12]
    has_missing_information = bool(assessment.missing_information)
    has_direct_support = bool(assessment.supporting_observation_ids) or assessment.recent_answer_supports
    covered = (
        assessment.information_need_resolved
        and has_direct_support
        and not has_missing_information
    )
    recap = assessment.recap_of_known_information and covered

    print(
        f"INQUIRY ASSESSMENT: {plan.thread_id}/{frontier.decision_key} | "
        f"covered={covered} | too_broad={assessment.too_broad} | "
        f"recap={recap} | move_on={assessment.should_move_on} | "
        f"current_value={assessment.current_frontier_value:.2f} | "
        f"alternative_value={assessment.best_alternative_value:.2f} | "
        f"higher_elsewhere={assessment.higher_value_elsewhere} | "
        f"repeats_rejected={assessment.repeats_rejected_frontier} | "
        f"repeats_prior={assessment.repeats_prior_decision} | "
        f"level={assessment.abstraction_level} | "
        f"material={assessment.material_product_consequence} | "
        f"missing={assessment.missing_information} | {assessment.reason}"
    )

    abstraction_reject = (
        assessment.abstraction_level in {"INTERACTION_DESIGN", "IMPLEMENTATION"}
        and not assessment.material_product_consequence
    )
    print("===== DISCOVERY ABSTRACTION DEBUG | STAGE 4: PYTHON CLASSIFICATION GATE =====")
    print(json.dumps({
        "abstraction_level": assessment.abstraction_level,
        "material_product_consequence": assessment.material_product_consequence,
        "should_move_on": assessment.should_move_on,
        "too_broad": assessment.too_broad,
        "covered": covered,
        "recap": recap,
        "higher_value_elsewhere": assessment.higher_value_elsewhere,
        "abstraction_rule_result": "REJECT" if abstraction_reject else "PASS",
        "abstraction_rule": (
            "Reject only when abstraction_level is INTERACTION_DESIGN or IMPLEMENTATION "
            "and material_product_consequence is false."
        ),
    }, ensure_ascii=False, indent=2, default=str))
    print("===== END STAGE 4 =====\n")

    if assessment.repeats_rejected_frontier:
        return (
            "The proposed frontier repeats an underlying decision that was already "
            "rejected during this planning cycle. Choose a materially different "
            "decision or thread."
        )
    if assessment.repeats_prior_decision:
        return (
            "The proposed frontier semantically repeats a decision already delivered "
            "in the interview, even though its ID/wording may differ. Prior question: "
            f"{assessment.matching_prior_question}. Choose a genuinely different "
            "product decision."
        )
    if (
        assessment.abstraction_level in {"INTERACTION_DESIGN", "IMPLEMENTATION"}
        and not assessment.material_product_consequence
    ):
        return (
            "The proposed frontier is below the required product-discovery abstraction "
            f"level ({assessment.abstraction_level}). The remaining detail is primarily "
            "UX/implementation refinement and does not materially change the product "
            "model or PRD. Move to a different product decision."
        )
    if covered or recap:
        support = (
            ", ".join(assessment.supporting_observation_ids)
            or ("the founder's latest contextual answer" if assessment.recent_answer_supports else "existing founder evidence")
        )
        return (
            "The proposed information need is already substantially answered or "
            f"is a recap of known information ({support}): {assessment.reason}"
        )
    if (
        assessment.higher_value_elsewhere
        and assessment.best_alternative_value > assessment.current_frontier_value
        and assessment.best_alternative_focus.strip()
    ):
        return (
            "BREADTH_PREFERENCE: This frontier is valid, but the assessor sees a "
            "potentially higher-value unresolved area elsewhere: "
            f"{assessment.best_alternative_focus}. This is a ranking preference, "
            "not a semantic rejection."
        )
    if assessment.should_move_on:
        alternative = (
            f" Possible next area: {assessment.best_alternative_focus}."
            if assessment.best_alternative_focus.strip() else ""
        )
        return (
            "Do not keep drilling this frontier. The proposed question itself has "
            "insufficient marginal discovery value at the current stage. "
            f"{assessment.depth_reason or assessment.reason}{alternative}"
        )
    if assessment.too_broad:
        return (
            "The proposed frontier bundles multiple product decisions or workflow "
            f"stages instead of one atomic uncertainty: {assessment.reason}"
        )
    return None


def _completion_sufficiency_problem(
    state: AgentState,
    scope: DiscoveryScope,
) -> str | None:
    """Return a minimal semantic reason discovery cannot safely finish yet.

    This is intentionally not a field checklist. It protects only foundational
    product-definition anchors whose absence makes a PRD describe an implementable
    guess rather than the founder's intended product.
    """
    knowledge = [
        item
        for item in state.get("discovered_knowledge", [])
        if item.scope == scope
        and item.knowledge_state == KnowledgeState.CONFIRMED
        and not item.absence
    ]

    has_actor_or_capability = any(
        item.topic == DiscoveryTopic.USER_ROLES
        and item.key in {"primary_users", "responsibilities"}
        for item in knowledge
    )
    has_explicit_outcome = any(
        item.topic == DiscoveryTopic.USER_GOALS
        and item.key in {"primary_user_goals", "secondary_user_goals", "motivations"}
        for item in knowledge
    )
    if has_actor_or_capability and not has_explicit_outcome:
        return (
            "DISCOVERY_INCOMPLETE: The product has confirmed users/capabilities but "
            "no explicit founder-provided user outcome or product goal. Ask one "
            "founder-facing question about the main outcome/job-to-be-done before "
            "considering PRD confirmation."
        )

    concepts = []
    for raw in state.get("product_concepts", []) or []:
        try:
            item = raw if isinstance(raw, ProductConcept) else ProductConcept.model_validate(raw)
        except Exception:
            continue
        if item.scope == scope:
            concepts.append(item)

    created_entity = any(
        item.key in {"responsibilities", "workflow_steps", "must_have_features"}
        and bool(re.search(r"\b(?:create|add|make|submit)\w*\b", f"{item.value} {item.evidence}", re.I))
        for item in knowledge
    )
    entities = {
        concept.subject.strip().lower()
        for concept in concepts
        if concept.kind == ProductConceptKind.ENTITY
    }
    has_non_state_attribute = any(
        concept.kind == ProductConceptKind.ATTRIBUTE
        and (concept.relation or "").strip().lower() not in {
            "status", "state", "lifecycle", "phase"
        }
        for concept in concepts
    )
    has_creation_validation = any(
        item.key == "validation_rules"
        or (
            item.key in {"responsibilities", "workflow_steps", "must_have_features"}
            and re.search(
                r"\b(?:field|title|name|description|required|optional|attribute|information)\w*\b",
                f"{item.value} {item.evidence}",
                re.I,
            )
        )
        for item in knowledge
    )
    if created_entity and entities and not (has_non_state_attribute or has_creation_validation):
        return (
            "DISCOVERY_INCOMPLETE: The founder can create/manage a core product "
            "entity, but the entity's usable information/shape has not been defined. "
            "Ask one product-level question about what information that created "
            "record/object needs to contain. Do not ask database/schema internals."
        )

    return None


def _plan_problem(
    plan: DiscoveryThreadPlan,
    state: AgentState,
    backlog: list[dict],
    rejected_frontiers: list[dict] | None = None,
) -> str | None:
    frontier = plan.frontier
    if (
        frontier is not None
        and plan.feedback is not None
        and plan.feedback.kind == ThreadFeedbackKind.PRODUCT_SCOPE_CLOSED
        and plan.thread_id == state.get("active_discovery_thread")
    ):
        return (
            "The founder explicitly closed the current line of product discovery. "
            "Do not continue inside the same thread; choose a materially different "
            "thread/decision or return no frontier if only requirements remain."
        )
    if frontier is not None:
        for rejected in rejected_frontiers or []:
            if (
                rejected.get("thread_id") == plan.thread_id
                and rejected.get("decision_key") == frontier.decision_key
            ):
                return (
                    "The proposed frontier exactly repeats a decision already rejected "
                    "during this planning cycle."
                )
        deliveries = sum(
            1
            for entry in state.get("requirement_question_history", [])[-12:]
            if entry.get("thread_id") == plan.thread_id
            and entry.get("decision_key") == frontier.decision_key
        )
        if deliveries >= 2:
            return (
                f"Decision '{frontier.decision_key}' in thread '{plan.thread_id}' "
                "has already been delivered twice."
            )
        if deliveries == 1 and state.get("extraction_status") != "NO_FACTS_FOUND":
            return (
                f"Decision '{frontier.decision_key}' already received a usable answer; "
                "advance to the next causal decision."
            )
        semantic_problem = _semantic_frontier_problem(
            plan,
            state,
            state.get("discovery_scope", DiscoveryScope.USER_APP),
            rejected_frontiers=rejected_frontiers,
        )
        if semantic_problem is not None:
            return semantic_problem
    if frontier is None:
        sufficiency_problem = _completion_sufficiency_problem(
            state,
            state.get("discovery_scope", DiscoveryScope.USER_APP),
        )
        if sufficiency_problem is not None:
            return sufficiency_problem

    if frontier is None and backlog and not plan.relevant_requirement_ids:
        return (
            "The plan has no model frontier and no relevant requirement, but eligible "
            "requirements still exist. Choose the next coherent thread/decision or mark "
            "at least one supplied requirement as relevant now."
        )
    return None


def _invoke_thread_plan(messages, *, repair: bool = False) -> DiscoveryThreadPlan:
    model = thread_planner_repair_model() if repair else thread_planner_model()
    result = model.invoke(messages)
    return (
        result
        if isinstance(result, DiscoveryThreadPlan)
        else DiscoveryThreadPlan.model_validate(result)
    )


def plan_discovery_thread(state: AgentState) -> DiscoveryThreadPlan:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    backlog = _requirement_payload(state, scope)
    payload = {
        "scope": scope.value,
        "raw_idea": (state.get("raw_idea", "") or "")[:1500],
        "latest_user_answer": _latest_human(state)[:2500],
        "latest_conversation_intent": state.get("conversation_intent"),
        "discovery_boundaries": [
            item for item in state.get("discovery_boundaries", [])[-16:]
            if not item.get("scope") or item.get("scope") == scope.value
        ],
        "recent_conversation": _recent_conversation(state),
        "new_confirmed_facts_this_turn": _latest_turn_facts(state, scope),
        "new_product_concepts_this_turn": _latest_turn_concepts(state, scope),
        "new_external_systems_this_turn": _latest_turn_external_systems(state, scope),
        "product_snapshot": _compact_product_snapshot(state, scope),
        "noncanonical_recent_evidence": _planner_observations(state, scope),
        "founder_gap_guidance": _gap_guidance_payload(state, scope),
        "current_threads": _compact_threads(state),
        "active_thread_id": state.get("active_discovery_thread"),
        "delivered_question_history": _history_payload(state),
        "thread_activity": _thread_activity_payload(state),
        "eligible_requirement_backlog": backlog[:12],
        "pending_frontiers": (state.get("deferred_discovery_frontiers", []) or [])[-8:],
    }
    rejected_frontiers: list[dict] = []
    deferred_frontiers: list[dict] = []
    last_problem = None
    captured_feedback = None
    soft_breadth_fallback = None
    soft_breadth_replan_used = False

    for attempt in range(3):
        attempt_payload = payload
        system_instruction = THREAD_PLANNER_INSTRUCTION
        if attempt:
            attempt_payload = {
                **payload,
                "rejected_frontiers": rejected_frontiers,
                "repair": {
                    "problem": last_problem,
                    "attempt": attempt,
                    "instruction": (
                        "Choose a materially different valid next move when the prior "
                        "problem is a breadth preference. Every item in rejected_frontiers "
                        "is semantically forbidden for this planning cycle: do not repeat "
                        "a HARD-rejected item with a new decision_key, narrower wording, "
                        "or a paraphrase. Choose EXACTLY ONE independently answerable "
                        "product decision. Stay on the current thread only when its NEXT "
                        "question is at least as valuable as the best unresolved alternative. "
                        "If problem starts with DISCOVERY_INCOMPLETE, discovery is "
                        "not allowed to finish: choose ONE founder-facing product decision "
                        "that directly resolves the named missing product-definition anchor. "
                        "Do not return frontier=null on that repair attempt. If the rejected "
                        "frontier was already covered, repeated in prior history, below the "
                        "product-decision abstraction level, or genuinely low-value under the "
                        "product-contract delta test, pause that thread and choose another "
                        "grounded decision or relevant requirement. Do not ask implementation/"
                        "UI mechanics, an end-to-end workflow recap, or another confirmation "
                        "of a closed answer. Use null for an uncertain anchor_gap and never "
                        "invent requirement IDs."
                    ),
                },
            }
            system_instruction += (
                "\nThis is a bounded repair attempt. Respect rejected_frontiers as "
                "semantic exclusions, not merely rejected wording."
            )

        proposed = None
        try:
            serialized_payload = json.dumps(
                attempt_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            print(
                "DISCOVERY PLANNER CONTEXT: "
                f"attempt={attempt + 1} "
                f"model_tier={'fast-repair' if attempt > 0 else 'reasoning'} "
                f"chars={len(serialized_payload):,} "
                f"rough_tokens~={max(1, len(serialized_payload) // 4):,}"
            )
            proposed = _normalize_plan(
                _invoke_thread_plan(
                    [
                        SystemMessage(content=system_instruction),
                        HumanMessage(content=serialized_payload),
                    ],
                    repair=attempt > 0,
                ),
                state,
                scope,
            )
            if proposed.feedback is not None and captured_feedback is None:
                captured_feedback = proposed.feedback
            problem = _plan_problem(
                proposed,
                state,
                backlog,
                rejected_frontiers=rejected_frontiers,
            )
            print("===== DISCOVERY ABSTRACTION DEBUG | STAGE 5: PLANNER VALIDATION RESULT =====")
            print(json.dumps({
                "attempt": attempt + 1,
                "thread_id": proposed.thread_id,
                "frontier": (
                    proposed.frontier.model_dump(mode="json")
                    if proposed.frontier is not None else None
                ),
                "validation_result": "ACCEPT" if problem is None else "REJECT_OR_REPLAN",
                "problem": problem,
            }, ensure_ascii=False, indent=2, default=str))
            print("===== END STAGE 5 =====\n")
            if problem is None:
                if proposed.feedback is None and captured_feedback is not None:
                    proposed = proposed.model_copy(update={"feedback": captured_feedback})
                return proposed.model_copy(
                    update={"deferred_frontiers": deferred_frontiers}
                )

            if problem.startswith("BREADTH_PREFERENCE:"):
                if soft_breadth_fallback is None:
                    soft_breadth_fallback = proposed
                    if proposed.frontier is not None:
                        deferred = {
                            **proposed.frontier.model_dump(mode="json"),
                            "scope": scope.value,
                            "thread_id": proposed.thread_id,
                            "thread_label": proposed.thread_label,
                            "thread_objective": proposed.thread_objective,
                        }
                        if not any(
                            item.get("thread_id") == deferred["thread_id"]
                            and item.get("decision_key") == deferred["decision_key"]
                            for item in deferred_frontiers
                        ):
                            deferred_frontiers.append(deferred)
                if soft_breadth_replan_used:
                    if proposed.feedback is None and captured_feedback is not None:
                        proposed = proposed.model_copy(update={"feedback": captured_feedback})
                    return proposed.model_copy(
                        update={"deferred_frontiers": deferred_frontiers}
                    )
                soft_breadth_replan_used = True
                last_problem = problem
                print(
                    "DISCOVERY THREAD BREADTH REPLAN: valid frontier deferred once "
                    "to give the planner a chance to choose a higher-value area"
                )
                continue

            last_problem = problem
        except Exception as exc:
            raise_if_llm_failure(exc)
            last_problem = f"Invalid structured thread plan: {exc}"

        print(
            f"DISCOVERY THREAD {'REPAIR' if attempt else 'REJECT'} "
            f"{attempt + 1}/3: {last_problem}"
        )

        if proposed is not None and proposed.frontier is not None:
            rejected = {
                "thread_id": proposed.thread_id,
                "decision_key": proposed.frontier.decision_key,
                "objective": proposed.frontier.objective,
                "question_hint": proposed.frontier.question_hint,
                "reason_rejected": last_problem,
            }
            if rejected not in rejected_frontiers:
                rejected_frontiers.append(rejected)

    if soft_breadth_fallback is not None:
        if soft_breadth_fallback.feedback is None and captured_feedback is not None:
            soft_breadth_fallback = soft_breadth_fallback.model_copy(
                update={"feedback": captured_feedback}
            )
        print(
            "DISCOVERY THREAD BREADTH FALLBACK: no better hard-valid frontier "
            "survived; using the previously valid deferred frontier"
        )
        selected_key = (
            soft_breadth_fallback.thread_id,
            soft_breadth_fallback.frontier.decision_key
            if soft_breadth_fallback.frontier is not None else None,
        )
        remaining_deferred = [
            item for item in deferred_frontiers
            if (item.get("thread_id"), item.get("decision_key")) != selected_key
        ]
        return soft_breadth_fallback.model_copy(
            update={"deferred_frontiers": remaining_deferred}
        )

    # Safe degradation: thread planning is a trajectory optimizer, not the
    # sole source of askable product inquiries. If every proposed thread frontier
    # is invalid, preserve the interview and hand control back to the existing
    # foundational/requirement inquiry pipeline instead of terminating the session.
    existing_threads = state.get("discovery_threads", {})
    closes_current_scope = (
        captured_feedback is not None
        and captured_feedback.kind == ThreadFeedbackKind.PRODUCT_SCOPE_CLOSED
    )
    fallback_thread_id = (
        "discovery_fallback"
        if closes_current_scope
        else (state.get("active_discovery_thread") or "discovery_fallback")
    )
    existing_thread = existing_threads.get(fallback_thread_id, {})
    print(
        "DISCOVERY THREAD FALLBACK: no valid model frontier survived bounded "
        "repair; delegating to foundational/requirement inquiries"
    )
    return DiscoveryThreadPlan(
        thread_id=fallback_thread_id,
        thread_label=existing_thread.get("label") or "Product discovery",
        thread_objective=(
            existing_thread.get("objective")
            or "Continue with the next unresolved high-value product decision."
        ),
        frontier=None,
        relevant_requirement_ids=[
            item["requirement_id"]
            for item in backlog
            if item.get("requirement_id")
        ],
        deferred_frontiers=deferred_frontiers,
        feedback=captured_feedback,
        rationale=(
            "No model-generated thread frontier survived semantic validation. "
            "Fallback to existing grounded inquiry and requirement candidates."
        ),
    )


def discovery_thread_node(state: AgentState) -> dict:
    if not state.get("thread_planning_enabled", False):
        return {}

    if state.get("founder_requested_completion"):
        # Founder closure is a strong stopping preference. Do not invent a fresh
        # model frontier after it. Existing foundational inquiries, validation
        # blockers, and active requirements will be arbitrated downstream.
        scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
        relevant_requirements = []
        for requirement in state.get("active_requirements", {}).values():
            if hasattr(requirement, "scope"):
                requirement_scope = requirement.scope
                requirement_status = getattr(requirement.status, "value", requirement.status)
                requirement_id = requirement.id
            else:
                requirement_scope = requirement.get("scope")
                requirement_status = requirement.get("status")
                requirement_id = requirement.get("id")
            if getattr(requirement_scope, "value", requirement_scope) != scope.value:
                continue
            if requirement_status != "ACTIVE":
                continue
            if requirement_id:
                relevant_requirements.append(requirement_id)
        return {
            "thread_frontier": None,
            "thread_relevant_requirement_ids": list(dict.fromkeys(relevant_requirements)),
            "completion_arbitration_complete": False,
            "question_retry_exhausted": False,
        }

    # Contradictions and explicit pending followups must be resolved before
    # ordinary conversational trajectory is reconsidered.
    if state.get("validation_candidate_blocking") or state.get("answer_followup"):
        return {
            "thread_frontier": None,
            "thread_relevant_requirement_ids": [],
        }

    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    plan = plan_discovery_thread(state)
    threads: Dict[str, dict] = dict(state.get("discovery_threads", {}))
    previous = state.get("active_discovery_thread")
    closes_previous = (
        plan.feedback is not None
        and plan.feedback.kind == ThreadFeedbackKind.PRODUCT_SCOPE_CLOSED
    )
    if previous and (previous != plan.thread_id or closes_previous) and previous in threads:
        threads[previous] = {
            **threads[previous],
            "status": ThreadStatus.PAUSED.value,
        }

    existing = threads.get(plan.thread_id, {})
    trigger_ids = list(dict.fromkeys([
        *(existing.get("trigger_fact_ids") or []),
        *(plan.frontier.related_fact_ids if plan.frontier else []),
    ]))
    thread = DiscoveryThread(
        id=plan.thread_id,
        label=plan.thread_label,
        objective=plan.thread_objective,
        scope=scope,
        parent_thread_id=plan.parent_thread_id,
        status=ThreadStatus.ACTIVE,
        trigger_fact_ids=trigger_ids,
        last_active_turn=state.get("turn_count", 0),
    )
    threads[plan.thread_id] = thread.model_dump(mode="json")

    frontier = None
    if plan.frontier is not None:
        frontier = {
            **plan.frontier.model_dump(mode="json"),
            "thread_id": plan.thread_id,
            "thread_label": plan.thread_label,
            "thread_objective": plan.thread_objective,
        }

    control_updates = {}
    latest_answer = _latest_human(state)
    if (
        plan.feedback is not None
        and plan.feedback.kind == ThreadFeedbackKind.DECISION_DEFERRED
        and plan.feedback.evidence
        and plan.feedback.evidence in latest_answer
    ):
        control_updates = apply_deferral(
            state,
            FreeTextDeferralReview(
                action="defer",
                primary_control_intent=True,
                kind=DeferralKind.DECISION,
                decision_summary=(
                    state.get("current_objective")
                    or (state.get("selected_inquiry") or {}).get("objective")
                    or plan.thread_objective
                ),
                reason=plan.feedback.instruction,
            ),
            latest_answer,
        )

    boundaries = list(
        control_updates.get(
            "discovery_boundaries",
            state.get("discovery_boundaries", []),
        )
    )
    if (
        plan.feedback is not None
        and plan.feedback.kind != ThreadFeedbackKind.DECISION_DEFERRED
    ):
        if plan.feedback.evidence and plan.feedback.evidence in latest_answer:
            boundary_type = {
                ThreadFeedbackKind.QUESTION_TOO_BROAD: "question_too_broad",
                ThreadFeedbackKind.IMPLEMENTATION_DEFERRED: "implementation_deferred",
                ThreadFeedbackKind.DECISION_DEFERRED: "decision_deferral",
                ThreadFeedbackKind.PRODUCT_SCOPE_CLOSED: "product_scope_closed",
            }[plan.feedback.kind]
            boundary = {
                "type": boundary_type,
                "scope": scope.value,
                "source_turn": state.get("turn_count", 0),
                "evidence": plan.feedback.evidence,
                "question": next(
                    (
                        message.content
                        for message in reversed(state.get("messages", [])[:-1])
                        if isinstance(message, AIMessage)
                    ),
                    "",
                ),
                "thread_id": previous or plan.thread_id,
                "decision_key": (state.get("selected_inquiry") or {}).get("decision_key"),
                "objective": state.get("current_objective"),
                "instruction": plan.feedback.instruction,
            }
            signature = (
                boundary["type"],
                boundary["source_turn"],
                boundary["evidence"],
            )
            existing_signatures = {
                (item.get("type"), item.get("source_turn"), item.get("evidence"))
                for item in boundaries
            }
            if signature not in existing_signatures:
                boundaries.append(boundary)

    deferred = list(state.get("deferred_discovery_frontiers", []) or [])
    selected = state.get("selected_inquiry") or {}
    receipt = state.get("active_answer_result") or {}
    answered_selected_frontier = bool(
        receipt
        and receipt.get("source_turn") == state.get("turn_count", 0)
        and receipt.get("directly_resolves", True)
        and selected.get("thread_id")
        and selected.get("decision_key")
    )
    feedback_closes_selected = (
        plan.feedback is not None
        and plan.feedback.kind in {
            ThreadFeedbackKind.DECISION_DEFERRED,
            ThreadFeedbackKind.IMPLEMENTATION_DEFERRED,
            ThreadFeedbackKind.PRODUCT_SCOPE_CLOSED,
        }
    )
    if answered_selected_frontier or feedback_closes_selected:
        deferred = [
            item for item in deferred
            if not (
                item.get("thread_id") == selected.get("thread_id")
                and item.get("decision_key") == selected.get("decision_key")
            )
        ]

    for item in plan.deferred_frontiers:
        if not any(
            existing.get("scope", scope.value) == item.get("scope", scope.value)
            and existing.get("thread_id") == item.get("thread_id")
            and existing.get("decision_key") == item.get("decision_key")
            for existing in deferred
        ):
            deferred.append(item)

    if plan.frontier is not None:
        deferred = [
            item for item in deferred
            if not (
                item.get("scope", scope.value) == scope.value
                and item.get("thread_id") == plan.thread_id
                and item.get("decision_key") == plan.frontier.decision_key
            )
        ]

    for boundary in boundaries:
        if boundary.get("type") not in {
            "decision_deferral",
            "implementation_deferred",
            "rejected_inquiry",
            "product_scope_closed",
        }:
            continue
        deferred = [
            item for item in deferred
            if not (
                (not boundary.get("scope") or boundary.get("scope") == item.get("scope", scope.value))
                and (
                    (
                        boundary.get("decision_key")
                        and boundary.get("decision_key") == item.get("decision_key")
                        and (
                            not boundary.get("thread_id")
                            or boundary.get("thread_id") == item.get("thread_id")
                        )
                    )
                    or (
                        boundary.get("objective")
                        and boundary.get("objective") == item.get("objective")
                    )
                )
            )
        ]

    return {
        **control_updates,
        "deferred_discovery_frontiers": deferred[-24:],
        "discovery_threads": threads,
        "active_discovery_thread": plan.thread_id,
        "thread_frontier": frontier,
        "thread_relevant_requirement_ids": list(plan.relevant_requirement_ids),
        "discovery_boundaries": boundaries[-50:],
        "question_retry_exhausted": False,
    }
