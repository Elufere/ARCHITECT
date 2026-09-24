"""Invalidation diagnostics identify committed evidence before status mutation."""
import builtins
import inspect
import json

from agents.topic_lifecycle import invalidate_completed_topics
from agents.state import DiscoveryTopic as T, TopicStatus
from test_topic_invalidation import actor, completed_state


def test_invalidation_identifies_fact_and_source_before_mutation(monkeypatch):
    initial = completed_state()
    vendor = actor(source_turn=14)
    messages = []

    def capture(message):
        if "=== TOPIC INVALIDATION ===" in message:
            # Inspect the local result mapping at the event boundary, not just
            # the input (which must remain unchanged throughout).
            caller = inspect.currentframe().f_back
            topic = caller.f_locals["topic"]
            assert caller.f_locals["updated"][topic] == TopicStatus.COMPLETED
            messages.append(message)

    monkeypatch.setattr(builtins, "print", capture)
    statuses = invalidate_completed_topics(initial,
        [*initial["discovered_knowledge"], vendor], [vendor], initial["topic_status"])
    assert len(messages) == 2
    ids = []
    for message in messages:
        assert "Old: COMPLETED\nNew: PARTIAL" in message
        assert "Source turn: 14" in message
        assert "reason: new confirmed same-scope actor vendor" in message
        trigger = next(line.removeprefix("Trigger: ") for line in message.splitlines() if line.startswith("Trigger: "))
        assert json.loads(trigger) == vendor.model_dump(mode="json")
        ids.append(next(line for line in message.splitlines() if line.startswith("Fact ID: sha256:")))
    assert ids[0] == ids[1]
    assert "new gaps: responsibilities::vendor, permissions::vendor" in messages[0]
    assert all(value == TopicStatus.COMPLETED for value in initial["topic_status"].values())
    assert all(value == TopicStatus.PARTIAL for value in statuses.values())


def test_multiple_triggers_have_distinct_stable_ids(capsys):
    initial = completed_state()
    facts = [actor("vendor", source_turn=14), actor("courier", source_turn=14)]
    ids = []
    for _ in range(2):
        invalidate_completed_topics(initial, [*initial["discovered_knowledge"], *facts],
                                    facts, initial["topic_status"])
        output = capsys.readouterr().out
        current = [line for line in output.splitlines() if line.startswith("Fact ID:")]
        assert len(set(current)) == 2
        ids.append(current)
    assert ids[0] == ids[1]


def test_no_event_without_invalidation(capsys):
    initial = completed_state()
    result = invalidate_completed_topics(initial, initial["discovered_knowledge"], [], initial["topic_status"])
    assert result == initial["topic_status"]
    assert capsys.readouterr().out == ""


def test_serialized_statuses_are_logged(capsys):
    initial = completed_state()
    vendor = actor(source_turn=14)
    statuses = {T.USER_ROLES.value: "COMPLETED"}
    result = invalidate_completed_topics(initial, [*initial["discovered_knowledge"], vendor], [vendor], statuses)
    assert result[T.USER_ROLES] == TopicStatus.PARTIAL
    assert "Topic: USER_ROLES\nOld: COMPLETED\nNew: PARTIAL" in capsys.readouterr().out
