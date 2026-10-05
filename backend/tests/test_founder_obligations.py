"""Regression coverage for founder obligations and semantic product concepts."""

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from agents import discovery_threads as threads
from agents import knowledge_tracker as tracker
from agents.discovery_obligations import (
    append_founder_obligations,
    open_founder_obligations,
)
from agents.inquiries import identify_open_inquiries
from agents.interview_planner import all_discovery_resolved
from agents.prd_projection import project_prd
from agents.prd_validation import build_source_snapshot
from agents.product_concepts import ProductConcept, ProductConceptKind
from agents.question_candidates import QuestionCandidate, filter_question_candidates
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem


def _obligation_state(description="how deletion should work"):
    base = {
        "discovery_scope": S.USER_APP,
        "turn_count": 2,
        "founder_obligations": [],
    }
    base["founder_obligations"] = append_founder_obligations(
        base,
        evidence=(
            "We still need to clarify what information a task contains, "
            "how deletion should work."
        ),
        items=[description],
    )
    return base


def test_founder_guidance_creates_durable_open_obligation_without_duplicates():
    state = _obligation_state()
    first = state["founder_obligations"]
    state["founder_obligations"] = append_founder_obligations(
        state,
        evidence="We still need to clarify how deletion should work.",
        items=["how deletion should work"],
    )

    assert len(first) == 1
    assert len(state["founder_obligations"]) == 1
    obligation = open_founder_obligations(state)[0]
    assert obligation.status == "OPEN"
    assert obligation.description == "how deletion should work"


def test_null_thread_frontier_cannot_hide_open_founder_obligation():
    state = {
        **_obligation_state(),
        "discovered_knowledge": [],
        "validation_issues": [],
        "validation_candidate_blocking": False,
        "answer_followup": None,
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "active_discovery_thread": "discovery_fallback",
        "thread_relevant_requirement_ids": [],
        "thread_frontier": None,
    }

    inquiries = identify_open_inquiries(state)

    assert len(inquiries) == 1
    assert inquiries[0].obligation_id
    assert inquiries[0].objective == "how deletion should work"


def test_open_obligation_is_not_repeat_suppressed_after_partial_answer():
    state = _obligation_state()
    obligation = open_founder_obligations(state)[0]
    inquiry_id = f"USER_APP|obligation|{obligation.id}"
    candidate = QuestionCandidate(
        id=f"question|{inquiry_id}",
        inquiry_id=inquiry_id,
        source="MODEL",
        scope=S.USER_APP,
        topic=T.CORE_WORKFLOW,
        objective=obligation.description,
        question_hint="Ask how deletion should work.",
        reason="Founder explicitly requested this decision.",
        thread_id="discovery_fallback",
        decision_key=f"founder_obligation.{obligation.id[-16:]}",
        obligation_id=obligation.id,
    )
    state.update({
        "question_candidates": [candidate.model_dump(mode="json")],
        "open_inquiries": [{
            "id": inquiry_id,
            "source": "MODEL",
            "scope": S.USER_APP.value,
            "topic": T.CORE_WORKFLOW.value,
            "objective": obligation.description,
            "question_hint": candidate.question_hint,
            "reason": candidate.reason,
            "thread_id": candidate.thread_id,
            "decision_key": candidate.decision_key,
            "obligation_id": obligation.id,
        }],
        "requirement_question_history": [{
            "inquiry_id": inquiry_id,
            "thread_id": candidate.thread_id,
            "decision_key": candidate.decision_key,
            "question": "Should deleted tasks be recoverable?",
            "turn": 3,
        }],
        "extraction_status": "SUCCESS",
    })

    eligible, decisions = filter_question_candidates(state, [candidate])

    assert [item.id for item in eligible] == [candidate.id]
    assert decisions[candidate.id].eligible is True


def test_completion_requires_no_open_obligations_and_no_planner_exhaustion():
    state = {
        **_obligation_state(),
        "open_inquiries": [],
        "active_requirements": {},
        "validation_issues": [],
        "validation_candidate_blocking": False,
        "answer_followup": None,
        "thread_plan_exit_reason": "STOPPING_TEST",
        "founder_requested_completion": False,
    }
    assert all_discovery_resolved(state) is False

    state["founder_obligations"] = []
    state["thread_plan_exit_reason"] = "EXHAUSTED"
    assert all_discovery_resolved(state) is False

    state["thread_plan_exit_reason"] = "STOPPING_TEST"
    assert all_discovery_resolved(state) is True


def test_crud_marginal_value_gate_cannot_veto_open_founder_obligation(monkeypatch):
    state = _obligation_state()
    obligation = open_founder_obligations(state)[0]
    state.update({
        "discovered_knowledge": [
            KnowledgeItem(
                topic=T.USER_ROLES,
                scope=S.USER_APP,
                key="primary_users",
                value="user",
                evidence="A user manages tasks.",
                roles=["user"],
                confidence=1,
            ),
            KnowledgeItem(
                topic=T.USER_ROLES,
                scope=S.USER_APP,
                key="responsibilities",
                value="delete tasks",
                evidence="A user can delete tasks.",
                role="user",
                confidence=1,
            ),
        ],
        "captured_observations": [],
        "founder_gap_guidance": [],
        "discovery_boundaries": [],
        "requirement_question_history": [],
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "discovery_threads": {},
        "messages": [],
        "turn_count": 3,
    })
    plan = threads.DiscoveryThreadPlan(
        thread_id="discovery_fallback",
        thread_label="Product discovery",
        thread_objective="Resolve founder-requested decisions.",
        frontier=threads.ThreadFrontierInquiry(
            decision_key="task_deletion_behavior",
            obligation_id=obligation.id,
            topic=T.CORE_WORKFLOW,
            objective="Clarify how task deletion should work.",
            question_hint="Should deleting a task permanently remove it or allow recovery?",
            reason="Founder explicitly requested deletion behavior.",
        ),
    )

    assessment = {
        "supporting_observation_ids": [],
        "recent_answer_supports": False,
        "information_need_resolved": False,
        "missing_information": ["task deletion behavior"],
        "too_broad": False,
        "recap_of_known_information": False,
        "should_move_on": True,
        "repeats_rejected_frontier": False,
        "repeats_prior_decision": False,
        "matching_prior_question": "",
        "abstraction_level": "PRODUCT_BEHAVIOR",
        "material_product_consequence": False,
        "current_frontier_value": 0.2,
        "best_alternative_value": 0.8,
        "higher_value_elsewhere": True,
        "best_alternative_focus": "another area",
        "depth_reason": "Ordinary CRUD depth.",
        "reason": "Normally low marginal value.",
    }
    monkeypatch.setattr(
        threads,
        "inquiry_assessment_model",
        lambda: SimpleNamespace(invoke=lambda _: assessment),
    )

    assert threads._semantic_frontier_problem(plan, state, S.USER_APP) is None


def test_extended_product_concepts_project_as_requirements_not_goals_or_global_scope():
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "product_concepts": [
            ProductConcept(
                kind=ProductConceptKind.PERSISTENCE,
                scope=S.USER_APP,
                subject="task",
                relation="storage",
                object="cloud",
                value="Tasks are stored in the cloud",
                evidence="Tasks should be stored in the cloud",
                confidence=1.0,
                source_turn=1,
            ),
            ProductConcept(
                kind=ProductConceptKind.BOUNDARY,
                scope=S.USER_APP,
                subject="task",
                relation="fields",
                object="closed_for_mvp",
                value="No other task fields are needed for the MVP",
                evidence="No other task fields are needed for the MVP",
                confidence=1.0,
                source_turn=1,
            ),
        ],
    }

    sources = build_source_snapshot(state)
    assert {source.key for source in sources} == {"persistence", "boundary"}

    projection = project_prd(sources)
    requirement_categories = {
        requirement.category for requirement in projection.draft.functional_requirements
    }
    assert "PRODUCT_MODEL.persistence" in requirement_categories
    assert "PRODUCT_MODEL.boundary" in requirement_categories
    assert projection.draft.elevator_pitch == []
    assert projection.draft.scope.out_of_scope == []



def test_todo_semantic_projection_keeps_meaning_in_correct_sections():
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [
            KnowledgeItem(
                topic=T.USER_ROLES,
                scope=S.USER_APP,
                key="primary_users",
                value="user",
                evidence="A user can manage tasks.",
                roles=["user"],
                confidence=1.0,
            ),
            KnowledgeItem(
                topic=T.USER_ROLES,
                scope=S.USER_APP,
                key="responsibilities",
                value="create tasks",
                evidence="A user can create tasks",
                role="user",
                confidence=1.0,
            ),
            KnowledgeItem(
                topic=T.USER_ROLES,
                scope=S.USER_APP,
                key="responsibilities",
                value="edit or delete tasks",
                evidence="edit or delete them",
                role="user",
                confidence=1.0,
            ),
            KnowledgeItem(
                topic=T.USER_GOALS,
                scope=S.USER_APP,
                key="primary_user_goals",
                value="access the same tasks across devices",
                evidence="so the user can access their tasks across devices",
                role="user",
                confidence=1.0,
            ),
            KnowledgeItem(
                topic=T.MVP_SCOPE,
                scope=S.USER_APP,
                key="out_of_scope",
                value="payments, integrations, or admin features",
                evidence="There are no other user roles, payments, integrations, or admin features",
                confidence=1.0,
            ),
        ],
        "product_concepts": [
            ProductConcept(
                kind=ProductConceptKind.ENTITY,
                scope=S.USER_APP,
                subject="task",
                value="Task",
                evidence="tasks",
                confidence=1.0,
                source_turn=1,
            ),
            ProductConcept(
                kind=ProductConceptKind.ATTRIBUTE,
                scope=S.USER_APP,
                subject="task",
                relation="has",
                object="title",
                value="Each task should have a title",
                evidence="Each task should have a title",
                confidence=1.0,
                source_turn=2,
            ),
            ProductConcept(
                kind=ProductConceptKind.STATE,
                scope=S.USER_APP,
                subject="task",
                relation="state",
                object="active or completed",
                value="Tasks can be either active or completed",
                evidence="Tasks can be either active or completed",
                confidence=1.0,
                source_turn=1,
            ),
            ProductConcept(
                kind=ProductConceptKind.OWNERSHIP,
                scope=S.USER_APP,
                subject="task",
                relation="belongs_to",
                object="user account",
                value="Tasks are tied to a user account",
                evidence="Tasks should be tied to a user account",
                confidence=1.0,
                source_turn=4,
            ),
            ProductConcept(
                kind=ProductConceptKind.PERSISTENCE,
                scope=S.USER_APP,
                subject="task",
                relation="storage",
                object="cloud",
                value="Tasks are stored in the cloud",
                evidence="stored in the cloud",
                confidence=1.0,
                source_turn=4,
            ),
            ProductConcept(
                kind=ProductConceptKind.BOUNDARY,
                scope=S.USER_APP,
                subject="task",
                relation="fields",
                object="closed_for_mvp",
                value="No other task fields are needed for the MVP",
                evidence="No other fields are needed for now",
                confidence=1.0,
                source_turn=2,
            ),
        ],
    }

    sources = build_source_snapshot(state)
    projection = project_prd(sources)
    elevator_categories = {claim.category for claim in projection.draft.elevator_pitch}
    out_scope_categories = {claim.category for claim in projection.draft.scope.out_of_scope}
    functional_categories = {
        requirement.category for requirement in projection.draft.functional_requirements
    }

    assert elevator_categories == {"USER_GOALS.primary_user_goals"}
    assert "PRODUCT_MODEL.persistence" in functional_categories
    assert "PRODUCT_MODEL.ownership" in functional_categories
    assert "PRODUCT_MODEL.boundary" in functional_categories
    assert "PRODUCT_MODEL.state" in functional_categories
    assert out_scope_categories == {"MVP_SCOPE.out_of_scope"}
    assert all(
        "cloud" not in claim.text.lower()
        for claim in projection.draft.elevator_pitch
    )
    assert all(
        "no other task fields" not in claim.text.lower()
        for claim in projection.draft.scope.out_of_scope
    )



def test_compound_storage_answer_can_capture_atomic_semantic_concepts(monkeypatch):
    text = (
        "Tasks should be tied to a user account and stored in the cloud so the "
        "user can access their tasks across devices."
    )
    items = [
        {
            "kind": "ownership_relationship",
            "subject": "task",
            "relation": "belongs_to",
            "object": "user account",
            "value": "Tasks are tied to a user account",
            "evidence": "Tasks should be tied to a user account",
            "confidence": 1.0,
            "knowledge_state": "CONFIRMED",
        },
        {
            "kind": "persistence_requirement",
            "subject": "task",
            "relation": "storage",
            "object": "cloud",
            "value": "Tasks are stored in the cloud",
            "evidence": "stored in the cloud",
            "confidence": 1.0,
            "knowledge_state": "CONFIRMED",
        },
        {
            "kind": "desired_outcome",
            "role": "user",
            "value": "access their tasks across devices",
            "evidence": "the user can access their tasks across devices",
            "confidence": 1.0,
            "knowledge_state": "CONFIRMED",
        },
    ]
    monkeypatch.setattr(
        tracker,
        "extraction_models",
        lambda: {"CLAIMS": SimpleNamespace(
            invoke=lambda _: {"parsed": {"items": items}}
        )},
    )

    state = {
        "messages": [HumanMessage(content=text)],
        "discovery_scope": S.USER_APP,
        "current_topic": T.BUSINESS_RULES,
        "current_gap": None,
        "discovered_knowledge": [
            KnowledgeItem(
                topic=T.USER_ROLES,
                scope=S.USER_APP,
                key="primary_users",
                value="user",
                evidence="A user manages tasks.",
                roles=["user"],
                confidence=1.0,
            )
        ],
        "turn_count": 4,
    }

    batch = tracker.extract_passes(text, state, S.USER_APP)

    concept_kinds = {concept.kind for concept in batch.concepts}
    assert ProductConceptKind.OWNERSHIP in concept_kinds
    assert ProductConceptKind.PERSISTENCE in concept_kinds
    goals = [item for item in batch if item.topic == T.USER_GOALS]
    assert len(goals) == 1
    assert goals[0].value == "access their tasks across devices"
    assert "cloud" not in goals[0].value.lower()
    assert "account" not in goals[0].value.lower()


def test_entity_local_no_more_fields_is_a_boundary_not_global_scope(monkeypatch):
    text = "No other fields are needed for now."
    items = [
        {
            "kind": "entity_boundary",
            "subject": "task",
            "relation": "fields",
            "object": "closed_for_mvp",
            "value": "No other task fields are needed for now",
            "evidence": text,
            "confidence": 1.0,
            "knowledge_state": "CONFIRMED",
        }
    ]
    monkeypatch.setattr(
        tracker,
        "extraction_models",
        lambda: {"CLAIMS": SimpleNamespace(
            invoke=lambda _: {"parsed": {"items": items}}
        )},
    )
    state = {
        "messages": [HumanMessage(content=text)],
        "discovery_scope": S.USER_APP,
        "current_topic": T.CORE_WORKFLOW,
        "current_gap": None,
        "discovered_knowledge": [],
        "turn_count": 2,
    }

    batch = tracker.extract_passes(text, state, S.USER_APP)

    assert not any(item.topic == T.MVP_SCOPE for item in batch)
    assert [concept.kind for concept in batch.concepts] == [
        ProductConceptKind.BOUNDARY
    ]
