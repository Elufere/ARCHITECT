"""Model-driven discovery reacts to knowledge changes through inquiries, not topic status."""

from copy import deepcopy

from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.inquiries import identify_open_inquiries
from agents.state import (
    DiscoveryScope as S,
    DiscoveryTopic as T,
    KnowledgeItem,
)


def fact(topic, key, value, **fields):
    return KnowledgeItem(
        topic=topic,
        scope=fields.pop("scope", S.USER_APP),
        key=key,
        value=value,
        evidence=value,
        confidence=1,
        **fields,
    )


def foundation():
    return [
        fact(T.USER_ROLES, "primary_users", "Customers use the app", roles=["customer"]),
        fact(T.USER_ROLES, "responsibilities", "Customers create orders", role="customer"),
        fact(T.USER_GOALS, "primary_user_goals", "Customers complete purchases", role="customer"),
        fact(T.CORE_WORKFLOW, "workflow_steps", "Create, pay, fulfil, complete"),
        fact(T.CORE_WORKFLOW, "completion_condition", "The transaction is completed"),
    ]


def actor(role="vendor"):
    return fact(
        T.USER_ROLES,
        "secondary_users",
        f"{role} uses the app",
        roles=[role],
    )


def test_new_actor_creates_new_model_inquiry_without_reopening_topics():
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [*foundation(), actor()],
        "fact_acquisition": {},
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "validation_issues": [],
        "validation_candidate_blocking": False,
    }

    inquiries = identify_open_inquiries(state)

    assert any(
        item.anchor_gap == "responsibilities::vendor"
        and item.role == "vendor"
        for item in inquiries
    )
    assert "topic_status" not in state
    assert "topic_maturity" not in state


def test_new_actor_with_actions_and_goal_does_not_create_redundant_actor_inquiry():
    vendor = actor()
    vendor_action = fact(
        T.USER_ROLES,
        "responsibilities",
        "Vendors fulfil orders",
        role="vendor",
    )
    vendor_goal = fact(
        T.USER_GOALS,
        "secondary_user_goals",
        "Vendors complete deliveries",
        role="vendor",
    )
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [*foundation(), vendor, vendor_action, vendor_goal],
        "fact_acquisition": {},
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "validation_issues": [],
        "validation_candidate_blocking": False,
    }

    inquiries = identify_open_inquiries(state)

    assert all(item.role != "vendor" for item in inquiries)


def test_incidental_non_actor_fact_does_not_create_topic_reopening_state():
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [
            *foundation(),
            fact(T.BUSINESS_RULES, "approval_rules", "A mediator approves disputes"),
        ],
        "fact_acquisition": {},
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "validation_issues": [],
        "validation_candidate_blocking": False,
    }

    identify_open_inquiries(state)

    assert "topic_status" not in state
    assert "topic_maturity" not in state


def test_knowledge_tracker_runtime_emits_no_topic_lifecycle_state(monkeypatch, capsys):
    text = "Vendors also use the app."
    initial = {
        "messages": [HumanMessage(content=text)],
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": foundation(),
        "superseded_knowledge": [],
        "fact_acquisition": {},
        "current_topic": None,
        "current_gap": None,
        "turn_count": 6,
    }
    candidate = actor()

    monkeypatch.setattr(tracker, "confirms_existing", lambda *_: False)
    monkeypatch.setattr(tracker, "interpret_closed_answer", lambda *_: None)
    monkeypatch.setattr(tracker, "extract_passes", lambda *_: [candidate])
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    monkeypatch.setattr(tracker, "ground_items", lambda items, *_: items)
    monkeypatch.setattr(
        tracker,
        "compare_candidate",
        lambda *_: ("new", None),
    )

    original = deepcopy(initial)
    result = tracker.knowledge_tracker_node(initial)

    assert initial == original
    assert "topic_status" not in result
    assert "topic_maturity" not in result
    output = capsys.readouterr().out
    assert "TOPIC STATUS" not in output
    assert "TOPIC INVALIDATION" not in output
