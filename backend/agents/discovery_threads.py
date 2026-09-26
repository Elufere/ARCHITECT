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
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator, model_validator

from agents.discovery_coverage import fact_id
from agents.llm import get_structured_model
from agents.llm_errors import ExtractionFailed, raise_if_llm_failure
from agents.state import AgentState, DiscoveryScope, DiscoveryTopic, KnowledgeState, TOPIC_KEY_MAP
from agents.product_concepts import ProductConcept


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


class DiscoveryThreadPlan(BaseModel):
    thread_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{1,79}$")
    thread_label: str
    thread_objective: str
    parent_thread_id: Optional[str] = None
    frontier: Optional[ThreadFrontierInquiry] = None
    relevant_requirement_ids: List[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("thread_id", "parent_thread_id", mode="before")
    @classmethod
    def normalize_thread_ids(cls, value):
        if value is None or not isinstance(value, str):
            return value
        normalized = re.sub(r"[^a-z0-9_.-]+", "_", value.strip().lower()).strip("_.-")
        return normalized[:80] or None


_thread_planner = None
_inquiry_assessment_model = None


class InquiryAssessment(BaseModel):
    """Semantic evidence audit for one proposed discovery frontier.

    The model reports evidence and missing information separately. Coverage is
    derived in code so "related context exists" cannot be mistaken for
    "the requested information is already known".
    """

    supporting_observation_ids: List[str] = Field(default_factory=list)
    recent_answer_supports: bool = False
    missing_information: List[str] = Field(default_factory=list)
    too_broad: bool
    recap_of_known_information: bool = False
    should_move_on: bool = False
    depth_reason: str = ""
    reason: str


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


THREAD_PLANNER_INSTRUCTION = """You are choosing the NEXT discovery move for a
product-manager interview. You do not create product facts. Confirmed facts are
authoritative; requirements are a backlog of decisions, not an interview agenda.

The interview should feel like an excellent human PM conversation:
1. Start with what CHANGED in the founder's latest answer. The payload explicitly
   identifies facts and product concepts captured on the latest turn. Follow the
   product structure, relationship, state, rule, or causal process that those new
   decisions just revealed.
2. Stay on one coherent discovery thread until the important local decisions are
   understandable. A child concept may temporarily become a child thread.
3. Prefer high-information forks that eliminate materially different product
   models. NEVER ask the founder to narrate an entire end-to-end workflow as one
   question when that workflow contains several distinct actors, stages, or
   decisions. Resolve one foundational product-shape fork or one causal link at a
   time. Ask the smallest question that will reshape the model, then follow its
   consequences. A question such as "walk me through the main steps from X to Y"
   is a recap/bundle, not a frontier decision, unless X->Y itself is one atomic
   transition whose rule is the unresolved decision.
   Examples of the reasoning pattern, NOT domain facts:
   - if a new entity appears, understand what it represents and how it relates
     to the current structure;
   - if a process step appears, understand the next unresolved causal link;
   - if a rule creates a consequence, ask about that consequence when its thread
     becomes relevant.
4. Do NOT mechanically collect actors' responsibilities, goals, permissions,
   exceptions, or every schema field before moving forward.
5. Do NOT jump from an incomplete normal/core flow into disputes, failures,
   edge cases, analytics, administration, or implementation merely because a
   globally important requirement exists. Defer it until its thread is active,
   unless it blocks the current decision or the founder explicitly made it the
   central subject.
6. Requirements may be relevant to the current thread. Return only supplied
   requirement IDs that should be eligible NOW. Leave unrelated requirements
   deferred; do not delete or resolve them.
7. Ask EXACTLY one product decision at a time. "One decision" means one
   independently answerable uncertainty. Do not bundle timing + process +
   conditions, permissions + features + experience, or actor identity +
   responsibilities into one frontier. If the founder could answer one part
   without answering another, they are separate decisions. Prefer the single
   highest-value one and leave the rest for later.
   WORKFLOW GRANULARITY IS STRICT:
   - Never ask for "the main actions", "the steps", "the flow", or "the journey"
     across a start-to-end range when several actions or decisions sit inside it.
   - Never ask multiple actors to describe their parts of the same workflow in one
     question.
   - A valid workflow frontier should normally be one actor + one stage/transition
     + one unresolved causal decision.
   - If the workflow is largely unknown, begin with the smallest useful entry
     point, such as the first action for one actor, rather than requesting the
     whole process.
8. Never repeat an underlying decision merely with different wording. The
   delivered-question history contains thread_id + decision_key, but wording and
   IDs are not the source of truth: use the founder's accumulated evidence too.
   If the substance was already answered across one or several earlier answers,
   move on. Never ask the founder to summarize or restate an already-known flow.
   If a decision was asked twice, choose another decision or another thread rather
   than paraphrasing it again.
9. The frontier is an UNCERTAINTY/DECISION, never an invented answer. Use the
   product's own vocabulary.
10. anchor_gap is optional normalization metadata only. Use null when no existing
    storage field cleanly represents the decision; never distort the question to
    fit a schema field.
11. Respect discovery_boundaries as persistent interview-control memory.
    - design_deferral means UI/interface/navigation/design implementation detail
      has been delegated away from the founder. Do not ask it again in different
      wording unless a concrete unresolved product decision genuinely depends on it.
    - rejected_inquiry means the founder rejected that inquiry as irrelevant or
      repeated. Do not retry, paraphrase, or deepen it. Move to a materially
      different product decision.
    These are NOT product facts and must never be converted into requirements.
12. If latest_conversation_intent is design_deferral or objection, the NEXT move
    must demonstrate that feedback was respected.
13. Read the founder's latest answer semantically, not only as product content.
    If it is primarily feedback about the INTERVIEW QUESTION itself — for example
    that the question asks for too much reasoning, too much of a process, or too
    many decisions at once — do not require a special phrase or intent label.
    Preserve the underlying product thread and replace the oversized inquiry with
    one smaller independently answerable decision. Do not store that feedback as
    product knowledge.

Choose a stable short thread_id and decision_key based on meaning, not wording.
Examples of generic thread shapes are core_interaction, checkout, fulfillment,
invitation, setup, settlement, access, but derive the actual thread from the
confirmed product model rather than copying these names.

Return frontier=null only when there is no model-level decision worth asking
before the currently relevant supplied requirements, or when discovery can
legitimately finish.
"""


def _latest_human(state: AgentState) -> str:
    return next(
        (
            message.content
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
    return result[-80:]


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
        result.append({
            "requirement_key": key,
            "requirement_id": requirement.id,
            "label": requirement.label,
            "description": requirement.description,
            "topic": requirement.topic.value,
            "parent_gap": requirement.parent_gap,
            "unresolved_facets": unresolved,
        })
    return result[:30]


def _recent_conversation(state: AgentState) -> list[dict]:
    result = []
    for message in state.get("messages", [])[-8:]:
        if isinstance(message, HumanMessage):
            result.append({"speaker": "founder", "text": message.content})
        elif isinstance(message, AIMessage):
            result.append({"speaker": "pm", "text": message.content})
    return result


def _history_payload(state: AgentState) -> list[dict]:
    result = []
    for entry in state.get("requirement_question_history", [])[-12:]:
        result.append({
            "turn": entry.get("turn"),
            "thread_id": entry.get("thread_id"),
            "decision_key": entry.get("decision_key"),
            "inquiry_id": entry.get("inquiry_id"),
            "question": entry.get("question"),
        })
    return result


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
    })


INQUIRY_ASSESSMENT_INSTRUCTION = """Assess the proposed next discovery
frontier before a question is generated.

Do NOT decide coverage from topical similarity. Separate what is KNOWN from what
the proposed frontier still asks the founder to supply.

Return:
- supporting_observation_ids: only observation IDs whose founder evidence
  DIRECTLY answers some or all of the proposed information need.
- recent_answer_supports=true when the immediately preceding founder answer
  directly resolves the proposed information need through conversational context,
  even if that short answer is not a stored observation.
- missing_information: each material part of the proposed information need that
  is NOT directly answered by existing founder evidence.
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
- should_move_on=true when the local decision is already sufficiently understood
  for product discovery and the proposed follow-up has low marginal value: e.g.
  it asks for UI/mechanism/implementation detail, repeatedly seeks an exhaustive
  list after the governing rule is clear, re-confirms an answer already closed
  by the founder, or continues drilling the same local subject while higher-level
  product structure remains unexplored. Set depth_reason to explain why.

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
Product discovery is not an exhaustive interrogation. Once the governing product
rule is clear, do not keep drilling merely because more implementation detail,
examples, UI mechanics, exhaustive enumeration, or edge specificity could exist.
If two or more recent questions have stayed on the same narrow decision and the
next answer would not materially change the product model, PRD decision, business
rule, architecture boundary, money movement, authorization model, or lifecycle,
set should_move_on=true. A founder request to avoid technical depth is also strong
evidence that implementation-level follow-ups should stop.

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
to a designer, or repeats/deepens an inquiry they rejected as irrelevant, set
should_move_on=true even if that detail remains unknown. Unknown does not mean
worth asking.

"""


def _semantic_frontier_problem(
    plan: DiscoveryThreadPlan,
    state: AgentState,
    scope: DiscoveryScope,
) -> str | None:
    frontier = plan.frontier
    if frontier is None:
        return None

    payload = {
        "proposed_frontier": {
            "thread_id": plan.thread_id,
            "decision_key": frontier.decision_key,
            "objective": frontier.objective,
            "question_hint": frontier.question_hint,
            "reason": frontier.reason,
        },
        "captured_founder_observations": _observation_payload(state, scope),
        "discovery_boundaries": [
            item for item in state.get("discovery_boundaries", [])[-50:]
            if not item.get("scope") or item.get("scope") == scope.value
        ],
        "latest_conversation_intent": state.get("conversation_intent"),
        "confirmed_product_facts": _fact_payload(state, scope),
        "confirmed_product_concepts": _concept_payload(state, scope),
        "recent_conversation": _recent_conversation(state),
        "delivered_question_history": _history_payload(state),
    }
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
    except Exception as exc:
        raise_if_llm_failure(exc)
        print(f"INQUIRY ASSESSMENT SKIPPED: {exc}")
        return None

    has_missing_information = bool(assessment.missing_information)
    has_direct_support = bool(assessment.supporting_observation_ids) or assessment.recent_answer_supports
    covered = has_direct_support and not has_missing_information
    recap = (
        assessment.recap_of_known_information
        and covered
    )

    print(
        f"INQUIRY ASSESSMENT: {plan.thread_id}/{frontier.decision_key} | "
        f"covered={covered} | too_broad={assessment.too_broad} | "
        f"recap={recap} | move_on={assessment.should_move_on} | "
        f"missing={assessment.missing_information} | {assessment.reason}"
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
    if assessment.should_move_on:
        return (
            "This local thread is sufficiently understood and the proposed follow-up "
            f"has low marginal discovery value: {assessment.depth_reason or assessment.reason}"
        )
    if assessment.too_broad:
        return (
            "The proposed frontier bundles multiple product decisions or workflow "
            f"stages instead of one atomic uncertainty: {assessment.reason}"
        )
    return None


def _plan_problem(
    plan: DiscoveryThreadPlan,
    state: AgentState,
    backlog: list[dict],
) -> str | None:
    frontier = plan.frontier
    if frontier is not None:
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
        )
        if semantic_problem is not None:
            return semantic_problem
    if frontier is None and backlog and not plan.relevant_requirement_ids:
        return (
            "The plan has no model frontier and no relevant requirement, but eligible "
            "requirements still exist. Choose the next coherent thread/decision or mark "
            "at least one supplied requirement as relevant now."
        )
    return None


def _invoke_thread_plan(messages) -> DiscoveryThreadPlan:
    result = thread_planner_model().invoke(messages)
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
        "raw_idea": state.get("raw_idea", ""),
        "latest_user_answer": _latest_human(state),
        "latest_conversation_intent": state.get("conversation_intent"),
        "discovery_boundaries": [
            item for item in state.get("discovery_boundaries", [])[-50:]
            if not item.get("scope") or item.get("scope") == scope.value
        ],
        "recent_conversation": _recent_conversation(state),
        "new_confirmed_facts_this_turn": _latest_turn_facts(state, scope),
        "new_product_concepts_this_turn": _latest_turn_concepts(state, scope),
        "confirmed_product_facts": _fact_payload(state, scope),
        "confirmed_product_concepts": _concept_payload(state, scope),
        "captured_observations": _observation_payload(state, scope),
        "current_threads": state.get("discovery_threads", {}),
        "active_thread_id": state.get("active_discovery_thread"),
        "delivered_question_history": _history_payload(state),
        "eligible_requirement_backlog": backlog,
    }
    messages = [
        SystemMessage(content=THREAD_PLANNER_INSTRUCTION),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]

    problem = None
    try:
        plan = _normalize_plan(_invoke_thread_plan(messages), state, scope)
        problem = _plan_problem(plan, state, backlog)
        if problem is None:
            return plan
    except Exception as first_exc:
        raise_if_llm_failure(first_exc)
        problem = f"Invalid structured thread plan: {first_exc}"

    print(f"DISCOVERY THREAD REPAIR: {problem}")
    repair_payload = {
        **payload,
        "repair": {
            "problem": problem,
            "instruction": (
                "Return one valid structured next move. Choose EXACTLY ONE independently "
                "answerable product decision. Do not combine timing, process, conditions, "
                "permissions, features, or experience into one frontier. Stay on the "
                "current causal/product-structure thread only when "
                "it still has material product value; if the repair says the local thread "
                "is sufficiently understood, move to a different high-value product "
                "decision. Do not ask for an end-to-end workflow recap, exhaustive list, "
                "implementation/UI mechanics, or another confirmation of a closed answer. "
                "Use null for an uncertain anchor_gap and do not invent requirement IDs."
            ),
        },
    }
    try:
        repaired = _normalize_plan(
            _invoke_thread_plan([
                SystemMessage(content=THREAD_PLANNER_INSTRUCTION + "\n"
                    "The previous plan was invalid or violated the conversation-control "
                    "protocol. Repair the plan only; do not add product facts."),
                HumanMessage(content=json.dumps(repair_payload, ensure_ascii=False)),
            ]),
            state,
            scope,
        )
        second_problem = _plan_problem(repaired, state, backlog)
        if second_problem is not None:
            raise ValueError(second_problem)
        return repaired
    except Exception as second_exc:
        raise_if_llm_failure(second_exc)
        raise ExtractionFailed("Discovery-thread planning failed after one repair attempt") from second_exc


def discovery_thread_node(state: AgentState) -> dict:
    if not state.get("thread_planning_enabled", False):
        return {}

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
    if previous and previous != plan.thread_id and previous in threads:
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

    return {
        "discovery_threads": threads,
        "active_discovery_thread": plan.thread_id,
        "thread_frontier": frontier,
        "thread_relevant_requirement_ids": list(plan.relevant_requirement_ids),
    }
