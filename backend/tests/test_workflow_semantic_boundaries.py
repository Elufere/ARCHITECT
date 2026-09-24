"""Workflow fields require their own process meaning, not the interview's focus.

Offline tests cover adversarial template filling and the independent absence
verdict. Set RUN_LIVE_WORKFLOW_BOUNDARIES=1 to check the production model too.
"""
import json
import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.state import DiscoveryScope as S, DiscoveryTopic as T


KEYS = ("trigger", "workflow_steps", "completion_condition", "downstream_dependency", "end_state")
NEGATIVES = [
    "Customers choose the app to reduce fraud risk.",
    "A user cannot be both buyer and seller in the same transaction.",
    "Users cannot switch roles during a transaction.",
    "Customers can open disputes.",
    "The buyer can fund the escrow.",
    "The interface is blue.",
    "A fraud team reviews suspicious payments.",
    "Only managers may approve requests.",
    "Accounts must be verified to access the dashboard.",
]
SINGLE_FACTS = [
    ("The review process starts when a customer submits a request.", "trigger"),
    ("The customer submits a request, a reviewer checks it, and then the customer receives a decision.", "workflow_steps"),
    ("After submission, the reviewer checks the request.", "workflow_steps"),
    ("The process is complete when all required checks have passed.", "completion_condition"),
    ("Before the workflow can proceed past verification, it requires a response from an external verifier.", "downstream_dependency"),
    ("After completion, the case status is archived.", "end_state"),
]


def state(text, gap):
    return dict(messages=[HumanMessage(content=text)], current_topic=T.CORE_WORKFLOW,
                current_gap=gap, discovery_scope=S.USER_APP, discovered_knowledge=[],
                topic_status={}, turn_count=1)


def install(monkeypatch, text, expected_keys, *, absence=None, confirmed_absence=False):
    calls = {}

    def invoke(name, messages):
        calls.setdefault(name, []).append(messages)
        if name == "WORKFLOW":
            prompt = messages[0].content
            assert "current CORE_WORKFLOW topic or gap is\nnot evidence" in prompt
            assert "A response may support exactly one workflow fact" in prompt
            assert "not a process-start event" in prompt
            assert "before proceeding or completing" in prompt
            # Simulate the model filling every field from the same sentence.
            return {"items": [dict(key=key, value="none" if absence else text,
                                   evidence=text, confidence=1,
                                   **({"absence": "none"} if absence else {}))
                              for key in (["downstream_dependency"] if absence else KEYS)]}
        if name == "GROUNDING":
            assert "For CORE_WORKFLOW, assess each field" in messages[0].content
            assert "exactly one supported fact is valid" in messages[0].content
            assert "denial of only one dependency type" in messages[0].content
            groups = json.loads(messages[-1].content)["evidence_groups"]
            candidates = [c for group in groups for c in group["candidates"]]
            return dict(
                evidence_categories=[dict(evidence_id=int(group["evidence_id"]),
                    categories=[f"CORE_WORKFLOW.{key}" for key in expected_keys]) for group in groups],
                # IDs alone cannot bypass category/absence checks.
                supported_ids=[c["id"] for c in candidates],
                confirmed_absence_ids=[c["id"] for c in candidates] if confirmed_absence else [],
                rejection_reasons={str(c["id"]): "Own quote does not support this workflow meaning"
                    for c in candidates if c["key"] not in expected_keys or (absence and not confirmed_absence)})
        return {"items": []}

    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name in [*(entry[0] for entry in tracker.PASSES), "GROUNDING"]})
    return calls


@pytest.mark.parametrize("gap", KEYS)
@pytest.mark.parametrize("text", NEGATIVES)
def test_nonprocess_statements_cannot_fill_fields_even_when_focused(monkeypatch, text, gap):
    calls = install(monkeypatch, text, [])
    result = tracker.knowledge_tracker_node(state(text, gap))
    assert result["discovered_knowledge"] == []
    assert "GROUNDING" in calls


@pytest.mark.parametrize("text,key", SINGLE_FACTS)
def test_single_supported_workflow_field_does_not_expand_to_a_template(monkeypatch, text, key):
    install(monkeypatch, text, [key])
    result = tracker.knowledge_tracker_node(state(text, "workflow_steps"))
    facts = result["discovered_knowledge"]
    assert len(facts) == 1 and facts[0].key == key
    assert facts[0].evidence == text and facts[0].absence is None


@pytest.mark.parametrize("text,categories", [
    ("Customers can open disputes.", []),
    ("The process requires no manual approval.", ["downstream_dependency"]),
])
def test_silence_and_partial_denial_do_not_establish_no_dependencies(monkeypatch, text, categories):
    install(monkeypatch, text, categories, absence=True)
    assert tracker.knowledge_tracker_node(state(text, "downstream_dependency"))["discovered_knowledge"] == []


def test_explicit_whole_field_dependency_absence_is_retained(monkeypatch):
    text = "This process has no downstream dependencies at all."
    install(monkeypatch, text, ["downstream_dependency"], absence=True, confirmed_absence=True)
    result = tracker.knowledge_tracker_node(state(text, "downstream_dependency"))
    assert len(result["discovered_knowledge"]) == 1
    absence = result["discovered_knowledge"][0]
    assert absence.key == "downstream_dependency" and absence.absence == "none"
    assert absence.evidence == text


@pytest.mark.skipif(os.environ.get("RUN_LIVE_WORKFLOW_BOUNDARIES") != "1",
                    reason="Requires the configured live Ollama model")
@pytest.mark.parametrize("text", NEGATIVES)
def test_live_nonprocess_statements(text):
    initial = state(text, "downstream_dependency")
    extracted = tracker.extract_passes(text, initial, S.USER_APP)
    assert not any(item.topic == T.CORE_WORKFLOW for item in extracted)


@pytest.mark.skipif(os.environ.get("RUN_LIVE_WORKFLOW_BOUNDARIES") != "1",
                    reason="Requires the configured live Ollama model")
@pytest.mark.parametrize("text,key", SINGLE_FACTS)
def test_live_single_field_workflow(text, key):
    initial = state(text, "workflow_steps")
    extracted = tracker.extract_passes(text, initial, S.USER_APP)
    grounded = tracker.ground_items(extracted, text, initial)
    for items in (extracted, grounded):
        workflow = [item for item in items if item.topic == T.CORE_WORKFLOW]
        assert {item.key for item in workflow} == {key}
        assert all(item.evidence in text for item in workflow)
