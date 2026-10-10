"""Offline regression for the exact To-do discovery session and PRD fallback."""
from types import SimpleNamespace

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from agents import discovery_threads as threads
from agents import knowledge_tracker
from agents.extraction_passes import NeutralClaim
from agents import pm_agent as pm
from agents.prd_semantics import (
    is_feature_local_rationale,
    is_system_behavior_claim,
    render_system_behavior,
)
from agents.product_concepts import ProductConcept, ProductConceptKind
from agents.prd_validation import PRDValidationError, claim_slots
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, KnowledgeState as K
from services.workspace import _prd_sections


RAW_IDEA = (
    "A simple personal to-do application that helps users organize and manage their daily tasks. "
    "Users can create tasks, edit task details, delete tasks, and mark tasks as completed. "
    "Tasks are organized into two states, active and completed, allowing users to track their "
    "progress and focus on unfinished work. The application is designed for individual use, "
    "with no additional user roles, payment features, third-party integrations, or administrative functionality."
)

TASK_FIELDS_QUESTION = (
    "I understand that users will be able to create, edit, delete, and complete tasks, which are "
    "organized into active and completed states. To shape the task details properly for the MVP, "
    "what specific information should each task include—should it be just a title, or also fields "
    "like a description, due date, or priority?"
)
REMINDER_QUESTION = (
    "Got it—each task must have a title, while description and due date/time are optional fields. "
    "To clarify how the app should handle due dates and times: when a user sets a due date and time "
    "for a task, should the app simply show that information, or should it also send a reminder to "
    "the user when the task is due?"
)
REOPEN_REMINDER_QUESTION = (
    "To clarify the product behavior: When a completed task is changed back to active and its due "
    "time is still in the future, should the app schedule again any reminder times that have not yet passed?"
)
OVERDUE_QUESTION = (
    "If an active task reaches its due date and time without being completed, should it remain active "
    "and be shown as overdue, or should some other status change occur?"
)


def _fact(topic, key, value, *, evidence=None, role=None, roles=None, absence=None,
          source_question=None, turn=0):
    return KnowledgeItem(
        topic=topic,
        scope=S.USER_APP,
        key=key,
        value=value,
        evidence=evidence or value,
        source_question=source_question,
        role=role,
        roles=roles,
        absence=absence,
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=turn,
    )


def _todo_session_state():
    knowledge = [
        _fact(T.USER_ROLES, "primary_users", "Users", evidence="Users", roles=["user"]),
        _fact(T.USER_ROLES, "responsibilities", "create tasks", role="user"),
        _fact(T.USER_ROLES, "responsibilities", "edit task details", role="user"),
        _fact(T.USER_ROLES, "responsibilities", "delete tasks", role="user"),
        _fact(T.USER_ROLES, "responsibilities", "mark tasks as completed", role="user"),
        _fact(T.MVP_SCOPE, "out_of_scope", "payment features"),
        _fact(T.MVP_SCOPE, "out_of_scope", "third-party integrations"),
        _fact(T.MVP_SCOPE, "out_of_scope", "administrative functionality"),
        _fact(T.USER_ROLES, "secondary_users", "none", evidence="no additional user roles",
              roles=[], absence="none"),
        _fact(T.BUSINESS_RULES, "validation_rules", "description should be optional",
              source_question=TASK_FIELDS_QUESTION, turn=1),
        _fact(T.BUSINESS_RULES, "validation_rules", "due date and time should be optional",
              source_question=TASK_FIELDS_QUESTION, turn=1),
        _fact(T.USER_ROLES, "responsibilities",
              "receives reminders 1hr, 30mins, and 5min before the due date and time",
              evidence="the app should send a reminder 1hr, 30mins, and 5min to the due date and time",
              role="user", source_question=REMINDER_QUESTION, turn=2),
        _fact(T.USER_ROLES, "responsibilities", "marks a task as completed before its due time",
              role="user", source_question=(
                  "If a user marks a task as completed before its due time, should all remaining "
                  "scheduled reminders for that task be canceled?"
              ), turn=3),
        _fact(T.CORE_WORKFLOW, "end_state",
              "all remaining scheduled reminders for that task are canceled",
              source_question=(
                  "If a user marks a task as completed before its due time, should all remaining "
                  "scheduled reminders for that task be canceled?"
              ), turn=3),
        _fact(T.USER_ROLES, "responsibilities", "change the status of a task from completed to active",
              role="user", source_question=(
                  "Should users be able to move a task from completed back to active?"
              ), turn=4),
        _fact(T.USER_ROLES, "responsibilities",
              "Automatically reschedule all future reminders that haven’t passed yet",
              role="user", source_question=REOPEN_REMINDER_QUESTION, turn=6),
        _fact(T.USER_GOALS, "primary_user_goals",
              "users get notified again as if the task was never completed",
              evidence="so users get notified again as if the task was never completed",
              role="user", source_question=REOPEN_REMINDER_QUESTION, turn=6),
        _fact(T.CORE_WORKFLOW, "workflow_steps", "requesting for deletion confirmation",
              source_question="Should deletion require confirmation before a task is removed?",
              turn=7),
        _fact(T.CORE_WORKFLOW, "end_state", "deleted task should be removed from the app",
              source_question="What should happen to task data after deletion?", turn=7),
        _fact(T.USER_ROLES, "permissions",
              "users will have to return the task to active before they can edit a completed task",
              role="user", turn=8),
        _fact(T.USER_ROLES, "permissions",
              "users are required to create an account before creating a task",
              role="user", turn=9),
        _fact(T.BUSINESS_RULES, "ownership_rules", "tasks are tied to the user's account", turn=11),
        _fact(T.CONSTRAINTS, "operational_constraints",
              "all devices show the same up-to-date task list", turn=11),
        _fact(T.MVP_SCOPE, "must_have_features",
              "support only mobile application for android and ios phones",
              evidence="It should support only mobile application for android and ios phones",
              turn=12),
    ]
    concepts = [
        ProductConcept(
            kind=ProductConceptKind.ENTITY, scope=S.USER_APP, subject="Tasks",
            value="Tasks", evidence="Tasks", confidence=1, source_turn=0,
        ),
        ProductConcept(
            kind=ProductConceptKind.ATTRIBUTE, scope=S.USER_APP, subject="Tasks",
            relation="are organized into", object="two states, active and completed",
            value="Tasks are organized into two states, active and completed",
            evidence="Tasks are organized into two states, active and completed",
            confidence=1, source_turn=0,
        ),
        ProductConcept(
            kind=ProductConceptKind.ATTRIBUTE, scope=S.USER_APP, subject="task",
            relation="has attribute", object="title",
            value="each task should include title", evidence="each task should include title",
            source_question=TASK_FIELDS_QUESTION, confidence=1, source_turn=1,
        ),
    ]
    overdue_frontier = {
        "decision_key": "unfinished-task-after-due-time",
        "topic": T.BUSINESS_RULES.value,
        "anchor_gap": None,
        "objective": (
            "Determine what happens to an active task when its due date and time pass without "
            "the task being completed."
        ),
        "question_hint": OVERDUE_QUESTION,
        "reason": (
            "The task lifecycle defines active and completed states and pre-due reminders, but "
            "does not define the state of an unfinished task after its deadline."
        ),
        "related_fact_ids": [],
        "information_gain": 0.88,
        "causal_relevance": 0.95,
        "conversation_continuity": 0.75,
        "architecture_impact": 0.8,
        "business_risk": 0.6,
        "question_cost": 0.0,
        "scope": S.USER_APP.value,
        "thread_id": "task-deadlines",
        "thread_label": "Due dates and reminders",
        "thread_objective": "Define the user-visible behavior associated with optional due dates and times.",
    }
    return {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": knowledge,
        "product_concepts": concepts,
        "external_systems": [],
        "discovery_boundaries": [],
        "deferred_discovery_frontiers": [overdue_frontier],
        "open_inquiries": [],
        "messages": [HumanMessage(content=RAW_IDEA)],
        "raw_idea": RAW_IDEA,
        "pm_is_complete": False,
    }


def test_feature_decision_rationale_is_not_promoted_to_global_product_goal():
    assert is_feature_local_rationale(
        "desired_outcome",
        "so users get notified again as if the task was never completed",
        REOPEN_REMINDER_QUESTION,
    )
    state = _todo_session_state()
    sources = pm.build_source_snapshot(state)
    projection = pm.project_prd(sources)

    assert not any(
        "notified again as if the task was never completed" in claim.text.lower()
        for claim in projection.draft.elevator_pitch
    )
    assert any(
        requirement.category == "USER_GOALS.primary_user_goals"
        and "notified again" in requirement.description.lower()
        for requirement in projection.draft.functional_requirements
    )


def test_app_behavior_answer_is_not_admitted_as_a_user_responsibility():
    state = _todo_session_state()
    answer = (
        "Automatically reschedule all future reminders that haven’t passed yet, "
        "so users get notified again as if the task was never completed."
    )
    state["messages"] = [
        AIMessage(content=REOPEN_REMINDER_QUESTION),
        HumanMessage(content=answer),
    ]
    claim = NeutralClaim(
        kind="actor_action",
        role="user",
        value="Automatically reschedule all future reminders that haven’t passed yet",
        evidence=answer,
        confidence=1,
        source_turn=6,
    )

    admitted = knowledge_tracker._admit_claim_item(
        claim, state, S.USER_APP, {"user"}, set()
    )

    assert admitted is not None
    assert admitted.topic == T.CORE_WORKFLOW
    assert admitted.key == "workflow_steps"
    assert admitted.role is None

    sources = pm.build_source_snapshot({
        **state,
        "discovered_knowledge": [*state["discovered_knowledge"], admitted],
    })
    projection = pm.project_prd(sources)
    assert any(
        requirement.category == "CORE_WORKFLOW.workflow_steps"
        and "When a completed task is returned to active" in requirement.description
        for requirement in projection.draft.functional_requirements
    )
    assert all(
        "The app sends reminders" not in behavior.text
        for persona in projection.draft.personas
        for behavior in persona.key_behaviors
    )


def test_feature_rationale_is_not_admitted_as_a_global_goal():
    state = _todo_session_state()
    state["messages"] = [
        AIMessage(content=REOPEN_REMINDER_QUESTION),
        HumanMessage(content=(
            "Automatically reschedule all future reminders that haven’t passed yet, "
            "so users get notified again as if the task was never completed."
        )),
    ]
    claim = NeutralClaim(
        kind="desired_outcome",
        role="user",
        value="users get notified again as if the task was never completed",
        evidence="so users get notified again as if the task was never completed",
        confidence=1,
        source_turn=6,
    )

    with pytest.raises(ValueError, match="Feature-specific decision rationale"):
        knowledge_tracker._admit_claim_item(
            claim, state, S.USER_APP, {"user"}, set()
        )


def test_system_behavior_is_not_rendered_as_a_user_responsibility():
    assert is_system_behavior_claim(
        REMINDER_QUESTION,
        "receives reminders 1hr, 30mins, and 5min before the due date and time",
    )
    state = _todo_session_state()
    sources = pm.build_source_snapshot(state)
    reminder_sources = [
        source for source in sources
        if source.key == "responsibilities"
        and "reminder" in source.value.lower()
    ]
    rendered = [render_system_behavior(source) for source in reminder_sources]
    assert any("The app sends reminders 1 hour, 30 minutes, and 5 minutes" in text for text in rendered)
    assert any("When a completed task is returned to active" in text for text in rendered)


def test_reopening_reminder_answer_does_not_resolve_due_date_edit_question(monkeypatch):
    state = _todo_session_state()
    frontier = threads.ThreadFrontierInquiry(
        decision_key="edited-due-date-reminders",
        topic=T.BUSINESS_RULES,
        objective="Determine how reminders behave when an existing task's due date and time are edited.",
        question_hint=(
            "When a user edits an existing task's due date and time, should future reminders "
            "whose scheduled times have not passed be rescheduled?"
        ),
        reason="Editing a due date is a distinct event from reopening a completed task.",
        related_fact_ids=[],
        information_gain=0.8,
        causal_relevance=0.9,
        conversation_continuity=0.8,
        architecture_impact=0.7,
        business_risk=0.5,
        question_cost=0.0,
    )
    plan = threads.DiscoveryThreadPlan(
        thread_id="task-deadlines",
        thread_label="Due dates and reminders",
        thread_objective="Define reminder lifecycle behavior.",
        frontier=frontier,
        relevant_requirement_ids=[],
    )
    false_positive_assessment = threads.InquiryAssessment(
        supporting_observation_ids=[],
        recent_answer_supports=True,
        information_need_resolved=True,
        missing_information=[],
        too_broad=False,
        recap_of_known_information=False,
        should_move_on=False,
        repeats_rejected_frontier=False,
        repeats_prior_decision=False,
        matching_prior_question="",
        abstraction_level="PRODUCT_BEHAVIOR",
        material_product_consequence=True,
        current_frontier_value=0.8,
        best_alternative_value=0.0,
        higher_value_elsewhere=False,
        best_alternative_focus="",
        depth_reason="",
        reason="Incorrectly assumes the reopening answer also covers editing a due date.",
    )
    monkeypatch.setattr(
        threads,
        "inquiry_assessment_model",
        lambda: SimpleNamespace(invoke=lambda _: false_positive_assessment),
    )

    assert threads._frontier_trigger_mismatch(plan, state, S.USER_APP)
    assert threads._semantic_frontier_problem(plan, state, S.USER_APP) is None


def test_open_questions_are_always_rendered_as_questions():
    questions = pm.build_open_questions(
        {
            "open_inquiries": [{
                "scope": S.USER_APP.value,
                "objective": "Determine whether active tasks become overdue",
                "question_hint": "Ask a focused question about overdue tasks",
            }],
            "deferred_discovery_frontiers": [],
            "discovery_boundaries": [],
        },
        S.USER_APP,
    )
    assert questions == ["Is it expected that active tasks become overdue?"]


def test_exact_todo_session_compiles_coherently_even_when_prose_polish_is_rejected(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)
    state = _todo_session_state()
    base_draft = pm.project_prd(pm.build_source_snapshot(state)).draft
    claim_id = next(iter(claim_slots(base_draft)))
    monkeypatch.setattr(
        pm,
        "prose_llm",
        SimpleNamespace(invoke=lambda _: {
            "parsed": {"edits": [{"claim_id": claim_id, "text": "A polished claim rejected by semantic audit."}]}
        }),
    )
    audit_calls = {"count": 0}

    def reject_polish_then_accept_fallback(*_args, **_kwargs):
        audit_calls["count"] += 1
        if audit_calls["count"] == 1:
            raise PRDValidationError("simulated source-category mismatch")

    monkeypatch.setattr(pm, "validate_prd", reject_polish_then_accept_fallback)

    result = pm.pm_compile_node(state)

    assert result["pm_is_complete"] is True
    assert audit_calls["count"] >= 2
    contract = result["prd_contract"]
    assert contract.prose_polished is False
    sections = {section.id: section for section in _prd_sections(contract, RAW_IDEA)}
    data_model = sections["data_model"].body.lower()
    assert "title" in data_model
    assert "description (optional)" in data_model
    assert "due date and time (optional)" in data_model

    feature_text = "\n".join(
        "\n".join([feature.title, feature.overview, *(detail.text for detail in feature.details)])
        for feature in contract.feature_specifications
    )
    assert "The app sends reminders 1 hour, 30 minutes, and 5 minutes" in feature_text
    assert "When a completed task is returned to active" in feature_text
    assert "Outcome: Users get notified again as if the task was never completed" in feature_text
    assert "Users can automatically reschedule" not in feature_text

    goals = sections.get("goals")
    assert goals is None or "users get notified again as if the task was never completed" not in goals.body.lower()
    assert OVERDUE_QUESTION in contract.open_questions
    assert "deleted task should be removed from the app" in sections["user_flow"].body.lower()
