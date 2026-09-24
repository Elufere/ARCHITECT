"""Full multi-turn escrow replay, with deterministic model boundary fixtures.

Extraction proposals and semantic verdicts are scripted (including adversarial
proposals); parsing, evidence checks, grounding enforcement, commits, deduplication,
absence supersession, coverage and lifecycle are real. This is not a live-model eval.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import knowledge_tracker as tracker
from agents.interview_planner import build_gap_info, interview_planner_node
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, TopicStatus as Status
from test_question_retry_limit import test_repeated_template_exits_graph_with_bounded_retries as check_retry_guardrail


def raw(key, value, evidence, **extra):
    return dict(key=key, value=value, evidence=evidence, confidence=1, **extra)


@pytest.fixture
def replay(monkeypatch):
    state = dict(messages=[], discovered_knowledge=[], topic_status={}, topic_maturity={},
                 discovery_scope=S.USER_APP, turn_count=0, current_topic=None)
    snapshots, logs, calls = {}, {}, []
    active = {}

    def invoke(name, messages):
        calls.append((active["name"], name))
        if name == "GAP_ANSWER":
            return dict(resolution="unresolved", evidence=active["text"], confidence=1)
        if name == "FACT_COMPARISON":
            return dict(relation="semantic_duplicate" if active["name"] == "repeat" else "new",
                        existing_id=0, confidence=1)
        if name == "CORRECTION_REVIEW":
            return dict(superseded_ids=[], confidence=1)
        if name == "GROUNDING":
            payload = json.loads(messages[-1].content)
            groups = payload["evidence_groups"]
            candidates = [c for g in groups for c in g["candidates"]]
            accepted = [c for c in candidates if c["value"] not in active.get("reject", set())]
            return dict(supported_ids=[c["id"] for c in accepted],
                evidence_categories=[dict(evidence_id=int(g["evidence_id"]), categories=[
                    f"{c['topic']}.{c['key']}" for c in g["candidates"] if c in accepted]) for g in groups],
                confirmed_absence_ids=[c["id"] for c in accepted if c.get("absence")],
                rejection_reasons={str(c["id"]): "Not supported by this statement" for c in candidates if c not in accepted})
        return dict(items=active["outputs"].get(name, []))

    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name in [*[p[0] for p in tracker.PASSES], "GAP_ANSWER", "GROUNDING", "FACT_COMPARISON", "CORRECTION_REVIEW"]})

    def turn(name, text, outputs, *, topic=T.USER_ROLES, gap=None, reject=()):
        active.update(name=name, text=text, outputs=outputs, reject=set(reject))
        state.update(current_topic=topic, current_gap=gap, turn_count=state["turn_count"] + 1)
        state["messages"] += [AIMessage(content="Please describe this part of the application."), HumanMessage(content=text)]
        out = StringIO()
        with redirect_stdout(out):
            state.update(tracker.knowledge_tracker_node(state))
        snapshots[name], logs[name] = deepcopy(state), out.getvalue()

    text = "the primary users are customers"
    turn("identity", text, {"ACTOR": [raw("primary_users", "customer", text, roles=["customer"])],
        "RESPONSIBILITY": [raw("responsibilities", "can browse items and pay", text, role="customer")]},
        reject=["can browse items and pay"])
    text = "we only have customers"
    turn("absence", text, {"ACTOR": [raw("secondary_users", "none", text + ".", roles=[], absence="none")]}, gap="secondary_users")
    text = "Customers can act as buyers or sellers depending on the transaction."
    turn("capacities", text, {"ACTOR": [raw("primary_users", text, text, roles=["customer"], aliases=["buyer", "seller"])]})
    text = "Customers may act as different roles across transactions, but never as buyer and seller in the same transaction. They cannot switch roles within a transaction."
    turn("policies", text, {"ACTOR": [raw("multiple_roles", text.split(". ")[0], text),
                                     raw("role_transitions", "cannot switch roles within a transaction", text)]})
    text = "Customers confirm completion. Only the customer may confirm their own transaction. Customers want payment protected until fulfillment. Success means fulfillment and payment release. Customers choose this app to reduce fraud risk."
    turn("coverage", text, {
        "RESPONSIBILITY": [raw("responsibilities", "customer confirms completion", text, role="customer")],
        "PERMISSION": [raw("permissions", "only customer may confirm their own transaction", text, role="customer")],
        "GOAL": [raw("primary_user_goals", "payment protected until fulfillment", text, role="customer"),
                 raw("success_criteria", "fulfillment and payment release", text),
                 raw("motivations", "reduce fraud risk", text)]})
    for topic in (T.USER_ROLES, T.USER_GOALS):
        assert not build_gap_info(state, topic)["missing_keys"]
        state["current_topic"] = topic
        with redirect_stdout(StringIO()):
            state.update(interview_planner_node(state))
        assert state["topic_status"][topic] == Status.COMPLETED
    snapshots["completed"] = deepcopy(state)
    text = "If there is a dispute, a mediator reviews the evidence."
    turn("participant", text, {"ACTOR": [raw("secondary_users", "mediator", text, roles=["mediator"])]}, reject=["mediator"])
    text = "Customers choose the app to reduce fraud risk."
    turn("motivation", text, {
        "GOAL": [raw("motivations", "reduce fraud risk", text)],
        "RESPONSIBILITY": [raw("responsibilities", "prevent fraud", text, role="customer")],
        "PERMISSION": [raw("permissions", "may prevent fraud", text, role="customer")],
        "WORKFLOW": [raw("trigger", "fraud concern starts transaction", text)]},
        reject=["prevent fraud", "may prevent fraud", "fraud concern starts transaction"])
    text = "The customer submits terms, then funds the transaction, then confirms completion."
    turn("workflow", text, {"WORKFLOW": [raw("workflow_steps", text, text)],
        "GOAL": [raw("primary_user_goals", "fund the transaction", text, role="customer")]},
        topic=T.CORE_WORKFLOW, reject=["fund the transaction"])
    text = "The customer must approve the transaction terms before funding."
    turn("approval", text, {"RULES": [raw("BUSINESS_RULES.approval_rules", text, text, topic="BUSINESS_RULES")]},
        topic=T.BUSINESS_RULES, gap="approval_rules")
    with redirect_stdout(StringIO()):
        snapshots["approval_planned"] = {**deepcopy(state), **interview_planner_node(deepcopy(state))}
    text = "The customer confirms the transaction is complete."
    turn("repeat", text, {"RESPONSIBILITY": [raw("responsibilities", "the customer confirms the transaction is complete", text, role="customer")]})
    text = "Actually, vendors also log into the same application and manage assigned orders."
    turn("new_actor", text, {"ACTOR": [raw("secondary_users", "vendor", text, roles=["vendor"])]})
    return snapshots, logs, calls


INVARIANTS = [
    "01_actor_without_invented_responsibilities", "02_explicit_secondary_absence",
    "03_contextual_capacities", "04_incidental_participant", "05_confirmed_new_user",
    "06_role_policies", "07_motivation_isolation", "08_workflow_not_goal",
    "09_canonical_rule_key", "10_no_duplicate_growth", "11_sticky_completion",
    "12_observable_reopening", "13_approval_gap_closed", "14_no_formatting_question_loop",
]


@pytest.mark.parametrize("invariant", INVARIANTS)
def test_escrow_failure_sequence(replay, invariant, monkeypatch):
    states, logs, calls = replay
    def items(step):
        return states[step]["discovered_knowledge"]
    def roles(step):
        return {r for i in items(step) for r in i.roles or []}
    number = int(invariant[:2])
    if number == 1:
        assert roles("identity") == {"customer"}
        assert not any(i.key == "responsibilities" for i in items("identity"))
    elif number == 2:
        absence = next(i for i in items("absence") if i.key == "secondary_users")
        assert absence.absence == "none" and absence.scope == S.USER_APP
        assert absence.evidence == "we only have customers"
    elif number == 3:
        assert roles("capacities") == {"customer"}
        assert any(i.aliases == {"customer": ["buyer", "seller"]} for i in items("capacities"))
    elif number == 4:
        assert roles("participant") == {"customer"}
        assert any(i.key == "secondary_users" and i.absence == "none" for i in items("participant"))
    elif number == 5:
        assert roles("new_actor") == {"customer", "vendor"}
        assert all(states["new_actor"]["topic_status"][t] == Status.PARTIAL for t in (T.USER_ROLES, T.USER_GOALS))
        assert not any(i.key == "secondary_users" and i.absence for i in items("new_actor"))
        assert states["new_actor"]["superseded_knowledge"]
    elif number == 6:
        policies = [i for i in items("policies") if i.key in ("multiple_roles", "role_transitions")]
        assert len(policies) == 2 and all(i.absence is None for i in policies)
        assert "across transactions" in policies[0].value and "never" in policies[0].value
        assert "within a transaction" in policies[1].value
    elif number == 7:
        assert items("motivation") == items("participant")
    elif number == 8:
        added = [i for i in items("workflow") if i not in items("motivation")]
        assert len(added) == 1 and added[0].key == "workflow_steps"
    elif number == 9:
        rule = next(i for i in items("approval") if i.key == "approval_rules")
        assert rule.topic == T.BUSINESS_RULES
        assert all("." not in i.key for i in items("approval"))
    elif number == 10:
        assert items("repeat") == items("approval")
        assert f"Knowledge count: {len(items('approval'))}" in logs["repeat"]
    elif number == 11:
        for step in ("participant", "motivation", "workflow", "approval", "repeat"):
            assert all(states[step]["topic_status"][t] == Status.COMPLETED for t in (T.USER_ROLES, T.USER_GOALS))
            assert "TOPIC INVALIDATION" not in logs[step]
    elif number == 12:
        event = logs["new_actor"]
        assert event.count("=== TOPIC INVALIDATION ===") == 2
        assert event.index("TOPIC STATUS MERGE") < event.index("=== TOPIC INVALIDATION ===")
        for label in ("Fact ID:", "Source turn:", "Old: COMPLETED", "New: PARTIAL", "responsibilities::vendor", "secondary_user_goals::vendor"):
            assert label in event
    elif number == 13:
        assert "approval_rules" not in build_gap_info(states["approval"], T.BUSINESS_RULES)["missing_keys"]
    elif number == 14:
        assert states["approval_planned"]["current_gap"] != "approval_rules"
        assert calls.count(("approval", "RULES")) == 1
        assert logs["approval"].count("FORMAT REPAIR ATTEMPT: 1/1") == 1
        # Keep the actual generation/guardrail graph bounded even when a model
        # insists on repeating its question. No guardrail is bypassed.
        check_retry_guardrail(monkeypatch)
