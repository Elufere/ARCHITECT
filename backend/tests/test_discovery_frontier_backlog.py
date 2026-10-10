"""Regression tests for breadth-deferred discovery frontiers."""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from agents import discovery_threads as threads
from agents.inquiries import identify_open_inquiries
from agents.state import DiscoveryScope as S, DiscoveryTopic as T


def _frontier(decision_key, objective, question, topic=T.BUSINESS_RULES):
    return threads.ThreadFrontierInquiry(
        decision_key=decision_key,
        topic=topic,
        objective=objective,
        question_hint=question,
        reason="This is a material product decision with a user-visible consequence.",
        related_fact_ids=[],
        information_gain=0.88,
        causal_relevance=0.95,
        conversation_continuity=0.75,
        architecture_impact=0.8,
        business_risk=0.6,
        question_cost=0.0,
    )


def _state():
    return {
        "messages": [
            AIMessage(content="Which platforms should the MVP support?"),
            HumanMessage(content="Android and iOS mobile apps only."),
        ],
        "raw_idea": "A personal to-do app.",
        "discovery_scope": S.USER_APP,
        "thread_planning_enabled": True,
        "discovered_knowledge": [],
        "product_concepts": [],
        "external_systems": [],
        "captured_observations": [],
        "founder_gap_guidance": [],
        "discovery_boundaries": [],
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "requirement_question_history": [],
        "discovery_threads": {},
        "deferred_discovery_frontiers": [],
        "active_discovery_thread": "task-deadlines",
        "thread_frontier": None,
        "thread_relevant_requirement_ids": [],
        "validation_issues": [],
        "validation_candidate_blocking": False,
        "answer_followup": None,
        "founder_requested_completion": False,
        "turn_count": 12,
    }


def test_valid_frontier_deferred_for_breadth_is_carried_into_next_plan(monkeypatch):
    overdue = threads.DiscoveryThreadPlan(
        thread_id="task-deadlines",
        thread_label="Due dates and reminders",
        thread_objective="Define due-date behavior.",
        frontier=_frontier(
            "unfinished-task-after-due-time",
            "Determine what happens to an active task after its due time passes.",
            "If an active task reaches its due date without completion, should it remain active and be shown as overdue, or change state?",
        ),
        relevant_requirement_ids=[],
    )
    platform = threads.DiscoveryThreadPlan(
        thread_id="platform-scope",
        thread_label="Supported platforms",
        thread_objective="Define the supported MVP platforms.",
        frontier=_frontier(
            "mvp-supported-platforms",
            "Determine which platforms the MVP supports.",
            "Which platforms should the MVP support?",
            topic=T.MVP_SCOPE,
        ),
        relevant_requirement_ids=[],
    )
    plans = iter([overdue, platform])
    monkeypatch.setattr(
        threads, "_invoke_thread_plan",
        lambda *_args, **_kwargs: next(plans),
    )
    monkeypatch.setattr(
        threads, "_plan_problem",
        lambda plan, *_args, **_kwargs: (
            "BREADTH_PREFERENCE: Supported platforms are higher priority."
            if plan.thread_id == "task-deadlines"
            else None
        ),
    )

    selected = threads.plan_discovery_thread(_state())

    assert selected.thread_id == "platform-scope"
    assert len(selected.deferred_frontiers) == 1
    assert selected.deferred_frontiers[0]["thread_id"] == "task-deadlines"
    assert selected.deferred_frontiers[0]["decision_key"] == "unfinished-task-after-due-time"


def test_deferred_frontier_is_persisted_and_reappears_as_an_open_inquiry(monkeypatch):
    state = _state()
    overdue = _frontier(
        "unfinished-task-after-due-time",
        "Determine what happens to an active task after its due time passes.",
        "If an active task reaches its due date without completion, should it remain active and be shown as overdue, or change state?",
    )
    overdue_record = {
        **overdue.model_dump(mode="json"),
        "scope": S.USER_APP.value,
        "thread_id": "task-deadlines",
        "thread_label": "Due dates and reminders",
        "thread_objective": "Define due-date behavior.",
    }
    plan = threads.DiscoveryThreadPlan(
        thread_id="platform-scope",
        thread_label="Supported platforms",
        thread_objective="Define the supported MVP platforms.",
        frontier=_frontier(
            "mvp-supported-platforms",
            "Determine which platforms the MVP supports.",
            "Which platforms should the MVP support?",
            topic=T.MVP_SCOPE,
        ),
        deferred_frontiers=[overdue_record],
    )
    monkeypatch.setattr(threads, "plan_discovery_thread", lambda _: plan)

    update = threads.discovery_thread_node(state)
    assert update["deferred_discovery_frontiers"][0]["decision_key"] == (
        "unfinished-task-after-due-time"
    )

    inquiry_state = {
        **state,
        **update,
        "active_discovery_thread": "platform-scope",
        "thread_frontier": update["thread_frontier"],
        "open_inquiries": [],
    }
    inquiries = identify_open_inquiries(inquiry_state)
    assert any(
        item.decision_key == "unfinished-task-after-due-time"
        for item in inquiries
    )

    closed_state = {
        **inquiry_state,
        "founder_requested_completion": True,
        "thread_frontier": None,
    }
    assert not identify_open_inquiries(closed_state)
