"""RULES serialization: canonical keys, strict pairs, and planner coverage."""
from typing import get_args

import pytest
from pydantic import ValidationError

from agents.extraction_passes import PASSES, RemainingFact
from agents.interview_planner import build_gap_info
from agents.state import DiscoveryTopic as T, TOPIC_KEY_MAP
from test_schema_alignment import FIELD_EXAMPLES, raw, run


TOPICS = [T(value) for value in get_args(RemainingFact.model_fields["topic"].annotation)]
FIELDS = [(topic, key) for topic in TOPICS for key in sorted(TOPIC_KEY_MAP[topic])]


@pytest.mark.parametrize("topic,key", FIELDS)
@pytest.mark.parametrize("prefixed", [False, True])
def test_all_remaining_fields_serialize_canonically(topic, key, prefixed):
    payload = raw(f"{topic.value}.{key}" if prefixed else key,
                  "explicit statement", "explicit statement", topic=topic.value)
    fact = RemainingFact.model_validate(payload)
    assert fact.model_dump()["key"] == key
    assert fact.model_dump()["topic"] == topic.value
    assert payload["key"] == (f"{topic.value}.{key}" if prefixed else key)
    assert RemainingFact.model_validate_json(fact.model_dump_json()) == fact


@pytest.mark.parametrize("topic", TOPICS)
def test_unknown_and_mismatched_keys_remain_invalid(topic):
    own_key = sorted(TOPIC_KEY_MAP[topic])[0]
    other = next(candidate for candidate in TOPICS if candidate != topic)
    other_key = sorted(TOPIC_KEY_MAP[other])[0]
    for key in ("unknown_field", f"{topic.value}.unknown_field", other_key,
                f"{topic.value}.{other_key}", f"{other.value}.{own_key}",
                f"{topic.value}.{topic.value}.{own_key}"):
        with pytest.raises(ValidationError):
            RemainingFact.model_validate(raw(key, "statement", "statement", topic=topic.value))


def test_schema_and_prompt_advertise_only_bare_keys():
    schema = RemainingFact.model_json_schema()
    assert set(schema["properties"]["key"]["enum"]) == {
        key for topic in TOPICS for key in TOPIC_KEY_MAP[topic]}
    assert "Never include a topic prefix" in schema["properties"]["key"]["description"]
    instruction = next(instruction for name, _, _, instruction in PASSES if name == "RULES")
    assert "emit key as the bare field name only" in instruction


@pytest.mark.parametrize("topic,key", FIELDS)
def test_prefixed_output_survives_commit_and_closes_its_gap(monkeypatch, topic, key):
    # Deterministic model outputs; real parsing, validation, grounding gate,
    # commit, and planner coverage. Includes approval_rules and invalid_actions.
    text = FIELD_EXAMPLES[topic][key]
    assert key in build_gap_info(dict(discovered_knowledge=[]), topic)["missing_keys"]
    state, calls = run(monkeypatch, text, {"RULES": [
        raw(f"{topic.value}.{key}", text, text, topic=topic.value)]},
        topic=topic, gap=key)
    records = state["discovered_knowledge"]
    assert len(records) == 1
    assert (records[0].topic, records[0].key, records[0].evidence) == (topic, key, text)
    assert key not in build_gap_info(state, topic)["missing_keys"]
    assert [name for name, _ in calls].count("RULES") == 1


def test_invalid_sibling_does_not_drop_valid_approval_rule(monkeypatch):
    text = FIELD_EXAMPLES[T.BUSINESS_RULES]["approval_rules"]
    state, _ = run(monkeypatch, text, {"RULES": [
        raw("BUSINESS_RULES.unknown_field", text, text, topic="BUSINESS_RULES"),
        raw("BUSINESS_RULES.approval_rules", text, text, topic="BUSINESS_RULES")]})
    assert [item.key for item in state["discovered_knowledge"]] == ["approval_rules"]
