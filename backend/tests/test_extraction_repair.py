"""Replay the live FixMate parse failure through the actual structured parser."""
import json
from types import SimpleNamespace
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from agents import knowledge_tracker as kt
from agents.interview_planner import build_gap_info
from agents.state import DiscoveryTopic as T, KnowledgeState as K

SOURCE = ('I want to build **FixMate**, a platform that helps people find and book trusted local '
          'artisans like electricians, plumbers, cleaners, painters, and appliance repair technicians. '
          'Users should be able to describe what they need, find suitable service providers nearby, '
          'compare them, book a service, communicate with the provider, and pay through the platform.')
QUOTE = SOURCE[SOURCE.index('Users should'):]


def item(key, value, **fields):
    return dict(topic='USER_ROLES', scope='USER_APP', key=key, value=value,
                evidence=QUOTE, confidence=1, knowledge_state='CONFIRMED',
                assertion_type='positive', roles=[], **fields)


def failed_batch():
    return {'items': [item('primary_users', 'users', aliases={}),
        item('secondary_users', 'service providers/artisans',
             aliases={'service_provider': ['service providers', 'artisans']}),
        item('multiple_roles', 'true')]}


def corrected_batch():
    customer = {**item('primary_users', 'users'), 'roles': ['users']}
    provider = {**item('secondary_users', 'service providers/artisans'),
                'evidence': SOURCE, 'roles': ['service_provider'],
                'aliases': {'service_provider': ['service providers', 'artisans']}}
    return {'items': [customer, provider]}


def replay(monkeypatch, payloads):
    parser = PydanticOutputParser(pydantic_object=kt.ExtractedKnowledge)
    calls = []
    def invoke(messages):
        payload = payloads[len(calls)]
        calls.append(messages)
        if isinstance(payload, Exception):
            raise payload
        return parser.parse(json.dumps(payload))
    monkeypatch.setattr(kt, 'extraction_llm', SimpleNamespace(invoke=invoke))
    result = kt.knowledge_tracker_node(dict(messages=[HumanMessage(content=SOURCE)],
        current_topic=None, current_gap=None, discovered_knowledge=[]))
    return result, calls


def test_live_failure_repaired_by_model_before_storage(monkeypatch):
    result, calls = replay(monkeypatch, [failed_batch(), corrected_batch()])
    assert len(calls) == 2
    # The installed Qwen template does not render later system messages.
    assert len(calls[1]) == 2 and isinstance(calls[1][-1], HumanMessage)
    assert SOURCE in calls[1][0].content
    assert 'Only an explicit absence' in calls[1][-1].content
    assert 'recheck evidence support for EVERY item' in calls[1][-1].content
    assert [i.roles for i in result['discovered_knowledge']] == [['users'], ['service_provider']]
    assert all(i.knowledge_state == K.CONFIRMED for i in result['discovered_knowledge'])
    assert not any(i.key == 'multiple_roles' for i in result['discovered_knowledge'])
    gaps = build_gap_info(result, T.USER_ROLES)['missing_keys']
    assert 'primary_users' not in gaps and 'secondary_users' not in gaps
    assert 'responsibilities::users' in gaps and 'responsibilities::service_provider' in gaps
    assert 'multiple_roles' in gaps


def test_second_invalid_batch_stops_without_salvaging_unsupported_sibling(monkeypatch):
    result, calls = replay(monkeypatch, [failed_batch(), failed_batch()])
    assert len(calls) == 2 and result == {}


def test_valid_output_does_not_retry(monkeypatch):
    result, calls = replay(monkeypatch, [corrected_batch()])
    assert len(calls) == 1 and len(result['discovered_knowledge']) == 2


def test_provider_failure_does_not_retry(monkeypatch):
    result, calls = replay(monkeypatch, [TimeoutError('provider timed out')])
    assert len(calls) == 1 and result == {}


def test_repaired_output_still_requires_exact_evidence(monkeypatch):
    repaired = corrected_batch()
    repaired['items'][0]['evidence'] = 'A fabricated source quote'
    result, calls = replay(monkeypatch, [failed_batch(), repaired])
    assert len(calls) == 2
    assert [i.roles for i in result['discovered_knowledge']] == [['service_provider']]


def test_roles_is_required_in_provider_schema():
    assert 'roles' in kt.ExtractedKnowledge.model_json_schema()['$defs']['ExtractedItem']['required']
