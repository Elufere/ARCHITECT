"""Contract -> production tracker -> production planner; no live model required."""
import logging
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from langchain_core.messages import HumanMessage
from agents import knowledge_tracker as kt, prior_gap_recovery as recovery
from agents.extraction_contract import ExtractedItem
from agents.interview_planner import interview_planner_node, build_gap_info
from agents.state import KnowledgeItem, KnowledgeState as K, DiscoveryTopic as T, DiscoveryScope as S

FIXMATE = ('The main users are customers who need services, service providers/artisans '
           'who offer the services, and admins who manage the platform.')


def fact(text, **fields):
    return dict(topic=T.USER_ROLES, scope=S.USER_APP, key='primary_users',
                value=text, evidence=text, confidence=1, knowledge_state=K.CONFIRMED,
                assertion_type='positive', roles=['customers'], **fields)


def actor(text, name, **fields):
    return {**fact(text), 'value': name, 'roles': [name], **fields}


def run(monkeypatch, text, items, gap='secondary_users', existing=()):
    prompts = []
    def invoke(messages):
        prompts.extend(messages)
        return kt.ExtractedKnowledge.model_validate({'items': items})
    monkeypatch.setattr(kt, 'extraction_llm', SimpleNamespace(invoke=invoke))
    state = dict(messages=[HumanMessage(content=text)], current_topic=T.USER_ROLES,
                 current_gap=gap, discovery_scope=S.USER_APP,
                 discovered_knowledge=list(existing), topic_status={}, turn_count=2)
    state.update(kt.knowledge_tracker_node(state))
    state.update(interview_planner_node(state))
    return state, prompts


def fixmate_items(text=FIXMATE):
    return [actor(text, 'customers'), actor(text, 'service_provider',
            aliases={'service_provider': ['service providers', 'artisan', 'artisans']}),
            actor(text, 'admin')]


@pytest.mark.parametrize('gap', ['secondary_users', 'primary_users', None, 'responsibilities::service_provider'])
def test_explicit_actors_and_all_six_gaps(monkeypatch, caplog, gap):
    caplog.set_level(logging.INFO)
    state, prompts = run(monkeypatch, FIXMATE, fixmate_items(), gap)
    records = state['discovered_knowledge']
    assert [(i.roles, i.knowledge_state) for i in records] == [
        ([r], K.CONFIRMED) for r in ['customers', 'service_provider', 'admin']]
    assert all(i.role is None for i in records)
    gaps = build_gap_info(state, T.USER_ROLES)['missing_keys']
    assert {f'{key}::{role}' for key in ['responsibilities', 'permissions']
            for role in ['customers', 'service_provider', 'admin']} <= set(gaps)
    assert not any('artisan' in gap for gap in gaps)
    for stage in ['LLM extracted state=CONFIRMED', 'validation=PASS',
                  'state after normalization=CONFIRMED', 'state after merge=CONFIRMED']:
        assert stage in caplog.text
    assert 'current_gap is conversational' in prompts[0].content


def test_explicit_extra_information_keeps_both_owners(monkeypatch):
    text = 'Providers accept bookings and complete jobs. Customers can cancel before the provider arrives.'
    existing = [ExtractedItem.model_validate(i) for i in fixmate_items()]
    items = [{**fact(text), 'key': 'responsibilities', 'roles': None, 'role': 'service_provider',
              'value': 'Providers accept bookings and complete jobs.'},
             {**fact(text), 'key': 'permissions', 'roles': None, 'role': 'customers',
              'assertion_type': 'conditional', 'value': 'Customers can cancel before the provider arrives.'}]
    state, _ = run(monkeypatch, text, items, 'responsibilities::service_provider', existing)
    provider, customer = state['discovered_knowledge'][-2:]
    assert provider.knowledge_state == customer.knowledge_state == K.CONFIRMED
    assert provider.role == 'service_provider' and customer.role == 'customers'
    assert customer.value == items[1]['value']


def test_genuine_inference_is_preserved_even_for_active_gap(monkeypatch):
    text = 'Someone from our team needs to manually approve every provider.'
    state, _ = run(monkeypatch, text, [actor(text, 'operator', key='secondary_users', knowledge_state=K.INFERRED)])
    assert state['discovered_knowledge'][0].knowledge_state == K.INFERRED
    assert not any('::operator' in g for g in build_gap_info(state, T.USER_ROLES)['missing_keys'])


@pytest.mark.parametrize('text,role,assertion', [
    ('There are no admins. Everything is automated.', 'admin', 'negative'),
    ('We may introduce delivery drivers next year.', 'driver', 'hypothetical'),
    ('We are unsure whether we need moderators.', 'moderator', 'uncertain'),
    ('If we enter logistics we will need drivers.', 'driver', 'conditional')])
def test_nonpositive_actor_is_not_current(monkeypatch, text, role, assertion):
    state, _ = run(monkeypatch, text, [actor(text, role, assertion_type=assertion)])
    assert state['discovered_knowledge'][0].assertion_type == assertion
    assert not any('::' in g for g in build_gap_info(state, T.USER_ROLES)['missing_keys'])


def test_alias_and_distinct_slash_relationships(monkeypatch):
    text = 'Customers book service providers/artisans.'
    state, _ = run(monkeypatch, text, fixmate_items(text)[:2])
    provider = state['discovered_knowledge'][1]
    assert provider.roles == ['service_provider'] and 'artisan' in provider.aliases['service_provider']
    text = 'Buyers/sellers are two separate actors: buyers pay and sellers receive payment.'
    state, _ = run(monkeypatch, text, [actor(text, 'buyer'), actor(text, 'seller')])
    assert [i.roles for i in state['discovered_knowledge']] == [['buyer'], ['seller']]
    assert all(not i.aliases for i in state['discovered_knowledge'])


def test_cross_role_contamination_is_impossible(monkeypatch):
    text = 'Customers book providers, and admins verify providers.'
    state, _ = run(monkeypatch, text, [actor(text, r) for r in ['customers', 'provider', 'admin']])
    assert len(state['discovered_knowledge']) == 3
    assert all(i.role is None for i in state['discovered_knowledge'])
    assert state['discovered_knowledge'][0].roles == ['customers']


@pytest.mark.parametrize('change', [dict(evidence='fabricated quote'), dict(confidence=.5),
    dict(role='provider'), dict(roles=['buyers/sellers']), dict(roles=['buyer', 'seller']),
    dict(key='not_a_key'), dict(assertion_type=None), dict(knowledge_state='GUESS'),
    dict(aliases={'unrelated': ['artisan']})])
def test_invalid_output_is_not_stored(monkeypatch, change):
    state, _ = run(monkeypatch, FIXMATE, [{**actor(FIXMATE, 'customers'), **change}])
    assert state['discovered_knowledge'] == []


def test_missing_provenance_rejected_but_legacy_loads():
    raw = actor(FIXMATE, 'customers')
    del raw['knowledge_state']
    del raw['assertion_type']
    legacy = KnowledgeItem.model_validate(raw)
    assert legacy.assertion_type is None
    with pytest.raises(ValidationError):
        kt.ExtractedKnowledge.model_validate({'items': [raw]})


def test_unknown_owner_and_conflicting_aliases_rejected(monkeypatch):
    text = 'Customers book providers.'
    item = {**fact(text), 'key': 'responsibilities', 'roles': None, 'role': 'unknown'}
    state, _ = run(monkeypatch, text, [item])
    assert state['discovered_knowledge'] == []
    items = [actor(text, 'customers', aliases={'customers': ['providers']}), actor(text, 'providers')]
    state, _ = run(monkeypatch, text, items)
    assert state['discovered_knowledge'] == []


def test_repeated_extraction_is_deduplicated(monkeypatch):
    state, _ = run(monkeypatch, FIXMATE, fixmate_items())
    again, _ = run(monkeypatch, FIXMATE, fixmate_items(), existing=state['discovered_knowledge'])
    assert len(again['discovered_knowledge']) == 3


def test_explicit_negative_update_removes_positive_actor_gaps(monkeypatch):
    existing = [ExtractedItem.model_validate(actor('There are admins.', 'admin'))]
    text = 'There are no admins. Everything is automated.'
    state, _ = run(monkeypatch, text, [actor(text, 'admin', value='There are no admins.', assertion_type='negative')], existing=existing)
    assert len(state['discovered_knowledge']) == 1
    assert not any('::admin' in gap for gap in build_gap_info(state, T.USER_ROLES)['missing_keys'])


def test_failed_correction_does_not_remove_existing_fact(monkeypatch):
    old = ExtractedItem.model_validate(actor('Someone reviews registrations.', 'operator', knowledge_state=K.INFERRED))
    monkeypatch.setattr(kt, 'extraction_llm', SimpleNamespace(invoke=lambda _: SimpleNamespace(
        items=[old.model_copy(update={'evidence': 'fabricated'})])))
    result = kt.knowledge_tracker_node(dict(messages=[HumanMessage(content='Actually, that is wrong.')],
        current_topic=T.USER_ROLES, current_gap='primary_users', is_correction=True,
        next_discovery_move='confirm_inference', discovered_knowledge=[old]))
    assert result['discovered_knowledge'] == [old]


def test_declared_alias_resolves_owner_without_value_rewrite(monkeypatch):
    existing = [ExtractedItem.model_validate(i) for i in fixmate_items()]
    text = 'Artisans accept bookings.'
    item = {**fact(text), 'key': 'responsibilities', 'roles': None, 'role': 'artisan'}
    state, _ = run(monkeypatch, text, [item], existing=existing)
    assert state['discovered_knowledge'][-1].role == 'service_provider'
    assert state['discovered_knowledge'][-1].value == text


@pytest.mark.parametrize('classification', [K.CONFIRMED, K.INFERRED])
def test_recovery_preserves_classification(monkeypatch, classification):
    text = 'Customers accept bookings.'
    source = ExtractedItem.model_validate(actor(text, 'customers'))
    payload = {**fact(text), 'key': 'responsibilities', 'roles': None, 'role': 'customers',
               'knowledge_state': classification}
    monkeypatch.setattr(recovery, 'recovery_model', lambda: SimpleNamespace(invoke=lambda _: recovery.RecoveryResult.model_validate(
        {'facts': [{'source_index': 0, 'item': payload}]})))
    result = recovery.recover_prior_gap(dict(current_topic=T.USER_ROLES, current_gap='responsibilities::customers',
                                           discovery_scope=S.USER_APP, discovered_knowledge=[source]))
    assert result['discovered_knowledge'][-1].knowledge_state == classification
