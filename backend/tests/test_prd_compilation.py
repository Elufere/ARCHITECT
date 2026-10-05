"""Deterministic PRD projection + optional prose polishing."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import pm_agent as pm
from agents.prd_projection import (
    apply_prose_edits,
    claim_slots,
    project_prd,
    validate_projection,
)
from agents.prd_specification import (
    build_feature_specifications,
    build_product_model,
    validate_feature_specifications,
)
from agents.prd_schema import (
    ClaimVerdict,
    PRDDraft,
    PRDProseEdit,
    SourceReference,
    SourcedClaim,
)
from agents.prd_validation import (
    PRDValidationError,
    build_source_snapshot,
    draft_claims,
    validate_prd,
)
from agents.product_concepts import ProductConcept, ProductConceptKind
from agents.state import (
    DiscoveryScope as S,
    DiscoveryTopic as T,
    KnowledgeItem,
    KnowledgeState as K,
)


QUOTE = "Managers approve orders over $100"


def fact(**changes):
    values = dict(
        topic=T.BUSINESS_RULES,
        scope=S.USER_APP,
        key="approval_rules",
        value="Orders over $100 require manager approval",
        evidence=QUOTE,
        role="manager",
        confidence=0.95,
        knowledge_state=K.CONFIRMED,
        source_turn=2,
    )
    values.update(changes)
    return KnowledgeItem(**values)


def state(*facts, scope=S.USER_APP):
    return dict(
        discovery_scope=scope,
        discovered_knowledge=list(facts or [fact()]),
        product_concepts=[],
        external_systems=[],
        discovery_boundaries=[],
        messages=[HumanMessage(content="UNTRUSTED_CHAT_ONLY")],
        raw_idea="UNTRUSTED_IDEA_ONLY",
        pm_is_complete=False,
    )


def install_prose(monkeypatch, parsed=None, raise_error=None):
    calls = []

    def invoke(messages):
        calls.append(messages)
        if raise_error is not None:
            raise raise_error
        return {"parsed": {"edits": []} if parsed is None else parsed}

    monkeypatch.setattr(pm, "prose_llm", SimpleNamespace(invoke=invoke))
    return calls


def all_true_verdict(claim_id="claim"):
    return ClaimVerdict(
        claim_id=claim_id,
        source_evidence_supports_facts=True,
        claim_supported=True,
        category_preserved=True,
        actors_preserved=True,
        conditions_preserved=True,
        validation_supported=True,
        no_conflict_with_confirmed_facts=True,
        explanation="Supported.",
    )


def test_source_ids_are_stable_and_corrections_get_new_ids():
    first = build_source_snapshot(state())[0]
    assert first.fact_id == build_source_snapshot(
        state(fact(confidence=1))
    )[0].fact_id
    assert first.fact_id != build_source_snapshot(
        state(fact(value="Orders over $200 require approval"))
    )[0].fact_id
    assert len(build_source_snapshot(state(fact(), fact()))) == 1


def test_short_answer_retains_its_original_question_in_snapshot():
    source = build_source_snapshot(
        state(
            fact(
                evidence="yes",
                source_question="Must managers approve orders over $100?",
            )
        )
    )[0]
    assert source.evidence == "yes"
    assert source.source_question == "Must managers approve orders over $100?"


def test_whole_field_absence_is_constraint_only_not_visible_prd_prose():
    primary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="user",
        evidence="A user manages tasks",
        roles=["user"],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    no_secondary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="secondary_users",
        value="none",
        evidence="There are no other user roles",
        roles=[],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
        absence="none",
    )
    sources = build_source_snapshot(state(primary, no_secondary))
    projection = project_prd(sources)
    validate_projection(projection, sources)

    primary_source = next(source for source in sources if source.key == "primary_users")
    absent_source = next(source for source in sources if source.key == "secondary_users")

    assert primary_source.fact_id in projection.visible_source_ids
    assert absent_source.fact_id in projection.constraint_source_ids
    assert len(projection.draft.personas) == 1
    assert projection.draft.personas[0].name == "User"
    assert absent_source.fact_id not in {
        ref
        for slot in claim_slots(projection.draft)
        for ref in slot["source_fact_ids"]
    }


def test_todo_sources_project_to_users_scope_and_functional_requirements():
    primary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="user",
        evidence="A user can create tasks",
        roles=["user"],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    no_secondary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="secondary_users",
        value="none",
        evidence="There are no other user roles",
        roles=[],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
        absence="none",
    )
    actions = [
        KnowledgeItem(
            topic=T.USER_ROLES,
            scope=S.USER_APP,
            key="responsibilities",
            value=value,
            evidence="A user can create, edit, delete and complete tasks",
            role="user",
            confidence=1,
            knowledge_state=K.CONFIRMED,
            source_turn=0,
        )
        for value in (
            "create tasks",
            "edit tasks",
            "delete tasks",
            "mark tasks as completed",
        )
    ]
    exclusions = [
        KnowledgeItem(
            topic=T.MVP_SCOPE,
            scope=S.USER_APP,
            key="out_of_scope",
            value=value,
            evidence="There are no payments, integrations, or admin features",
            confidence=1,
            knowledge_state=K.CONFIRMED,
            source_turn=0,
        )
        for value in ("payments", "integrations", "admin features")
    ]
    concept = ProductConcept(
        kind=ProductConceptKind.ENTITY,
        scope=S.USER_APP,
        subject="task",
        value="Tasks can be either active or completed",
        evidence="Tasks can be either active or completed",
        confidence=1,
        source_turn=0,
    )
    initial = state(primary, no_secondary, *actions, *exclusions)
    initial["product_concepts"] = [concept.model_dump(mode="json")]

    sources = build_source_snapshot(initial)
    projection = project_prd(sources)
    validate_projection(projection, sources)

    assert [persona.name for persona in projection.draft.personas] == ["User"]
    assert {item.text for item in projection.draft.scope.out_of_scope} == {
        "payments",
        "integrations",
        "admin features",
    }
    descriptions = {
        item.description for item in projection.draft.functional_requirements
    }
    assert {
        "create tasks",
        "edit tasks",
        "delete tasks",
        "mark tasks as completed",
        "Tasks can be either active or completed",
    }.issubset(descriptions)
    assert len(projection.constraint_source_ids) == 1


def test_product_concepts_are_first_class_projection_sources():
    concept = ProductConcept(
        kind=ProductConceptKind.ATTRIBUTE,
        scope=S.USER_APP,
        subject="task",
        relation="has attribute",
        object="required title",
        value="Each task should have a required title",
        evidence="Each task should have a required title",
        source_question="What information should a task contain?",
        confidence=1,
        source_turn=3,
    )
    initial = state()
    initial["discovered_knowledge"] = []
    initial["product_concepts"] = [concept.model_dump(mode="json")]

    sources = build_source_snapshot(initial)
    projection = project_prd(sources)
    validate_projection(projection, sources)

    assert len(sources) == 1
    assert sources[0].source_question == "What information should a task contain?"
    assert projection.draft.functional_requirements[0].category == (
        "PRODUCT_MODEL.attribute"
    )
    assert projection.draft.functional_requirements[0].source_fact_ids == [
        sources[0].fact_id
    ]


def test_deterministic_projection_saves_even_when_prose_model_returns_nothing(
    monkeypatch,
    tmp_path,
):
    install_prose(monkeypatch, parsed={"edits": []})
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)

    result = pm.pm_compile_node(state())

    assert result["pm_is_complete"] is True
    assert result["compilation_errors"] == []
    assert result["prd_contract"].prose_polished is False
    saved = json.loads((tmp_path / "requirements_mvp.json").read_text())
    assert saved["functional_requirements"][0]["description"] == (
        "Orders over $100 require manager approval"
    )
    assert saved["prose_polished"] is False


def test_bad_or_unavailable_prose_model_falls_back_to_projection(
    monkeypatch,
    tmp_path,
):
    install_prose(monkeypatch, raise_error=RuntimeError("formatter unavailable"))
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)

    result = pm.pm_compile_node(state())

    assert result["pm_is_complete"] is True
    assert result["prd_contract"].prose_polished is False
    assert (tmp_path / "requirements_mvp.json").exists()


def test_unknown_prose_claim_id_cannot_mutate_or_block_prd(
    monkeypatch,
    tmp_path,
):
    install_prose(
        monkeypatch,
        parsed={
            "edits": [
                {
                    "claim_id": "invented/claim",
                    "text": "Invent a new requirement",
                    "validation": None,
                }
            ]
        },
    )
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)

    result = pm.pm_compile_node(state())

    assert result["pm_is_complete"] is True
    assert result["prd_contract"].prose_polished is False
    assert result["prd_contract"].functional_requirements[0].description == (
        "Orders over $100 require manager approval"
    )


def test_semantically_rejected_polish_reverts_to_grounded_projection(
    monkeypatch,
    tmp_path,
):
    install_prose(
        monkeypatch,
        parsed={
            "edits": [
                {
                    "claim_id": "functional_requirements/FR-01",
                    "text": "Only managers can see orders over $100.",
                    "validation": None,
                }
            ]
        },
    )
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)

    def reject(*_args, **_kwargs):
        raise PRDValidationError("visibility was not supported")

    monkeypatch.setattr(pm, "validate_prd", reject)

    result = pm.pm_compile_node(state())

    assert result["pm_is_complete"] is True
    assert result["prd_contract"].prose_polished is False
    assert result["prd_contract"].functional_requirements[0].description == (
        "Orders over $100 require manager approval"
    )


def test_supported_prose_edit_is_applied_without_changing_structure(
    monkeypatch,
    tmp_path,
):
    install_prose(
        monkeypatch,
        parsed={
            "edits": [
                {
                    "claim_id": "functional_requirements/FR-01",
                    "text": "Orders above $100 require manager approval.",
                    "validation": None,
                }
            ]
        },
    )
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pm, "validate_prd", lambda *_args, **_kwargs: [])

    result = pm.pm_compile_node(state())

    requirement = result["prd_contract"].functional_requirements[0]
    source = result["prd_contract"].source_facts[0]
    assert result["pm_is_complete"] is True
    assert result["prd_contract"].prose_polished is True
    assert requirement.description == "Orders above $100 require manager approval."
    assert requirement.category == "BUSINESS_RULES.approval_rules"
    assert requirement.source_fact_ids == [source.fact_id]
    assert requirement.id == "FR-01"


def test_apply_prose_edits_cannot_change_categories_ids_or_sources():
    sources = build_source_snapshot(state())
    projection = project_prd(sources)
    original = projection.draft.functional_requirements[0]

    edited = apply_prose_edits(
        projection.draft,
        [
            PRDProseEdit(
                claim_id="functional_requirements/FR-01",
                text="Clearer wording",
            )
        ],
    )
    changed = edited.functional_requirements[0]

    assert changed.description == "Clearer wording"
    assert changed.id == original.id
    assert changed.category == original.category
    assert changed.actor_ids == original.actor_ids
    assert changed.source_fact_ids == original.source_fact_ids


def test_missing_evidence_or_empty_snapshot_blocks_projection_before_polish(
    monkeypatch,
    tmp_path,
):
    calls = install_prose(monkeypatch)
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)

    assert not pm.pm_compile_node(state(fact(evidence="")))["pm_is_complete"]
    assert not pm.pm_compile_node(
        state(fact(knowledge_state=K.INFERRED))
    )["pm_is_complete"]
    assert calls == []


def test_oversized_optional_polish_payload_falls_back_without_losing_prd(
    monkeypatch,
    tmp_path,
):
    calls = install_prose(monkeypatch)
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)
    initial = state(fact(evidence="x" * 100000))

    result = pm.pm_compile_node(initial)

    assert result["pm_is_complete"] is True
    assert result["prd_contract"].prose_polished is False
    assert calls == []


def test_admin_scope_has_separate_artifact(monkeypatch, tmp_path):
    install_prose(monkeypatch)
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)

    result = pm.pm_compile_node(
        state(fact(scope=S.ADMIN_DASHBOARD), scope=S.ADMIN_DASHBOARD)
    )

    assert result["pm_is_complete"] is True
    assert (tmp_path / "requirements_admin_dashboard.json").exists()
    assert not (tmp_path / "requirements_mvp.json").exists()


def test_failed_atomic_replace_keeps_previous_prd(monkeypatch, tmp_path):
    install_prose(monkeypatch)
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)
    path = tmp_path / "requirements_mvp.json"
    path.write_text("PREVIOUS_VERIFIED_PRD", encoding="utf-8")

    def fail(*_):
        raise OSError("disk error")

    monkeypatch.setattr(pm.os, "replace", fail)
    result = pm.pm_compile_node(state())

    assert not result["pm_is_complete"]
    assert path.read_text() == "PREVIOUS_VERIFIED_PRD"
    assert not list(tmp_path.glob(".prd-*.tmp"))


def test_persistent_founder_deferral_survives_projected_prd(monkeypatch, tmp_path):
    install_prose(monkeypatch)
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)
    initial = state()
    initial["discovery_boundaries"] = [
        {
            "id": "deferral:fee-policy",
            "type": "decision_deferral",
            "kind": "decision",
            "scope": S.USER_APP.value,
            "source_turn": 7,
            "evidence": "Let's decide the fee cap later.",
            "decision_summary": "Maximum transaction fee cap",
            "resolution_stage": "later",
            "requirement_id": "fee_policy",
            "explicit_authorization": True,
        }
    ]

    result = pm.pm_compile_node(initial)

    assert result["pm_is_complete"] is True
    deferred = result["prd_contract"].deferred_decisions
    assert len(deferred) == 1
    assert deferred[0].decision == "Maximum transaction fee cap"
    assert deferred[0].evidence == "Let's decide the fee cap later."


def test_reopened_deferral_is_not_published(monkeypatch, tmp_path):
    install_prose(monkeypatch)
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)
    initial = state()
    initial["discovery_boundaries"] = [
        {
            "id": "deferral:fees",
            "type": "decision_deferral",
            "kind": "decision",
            "scope": S.USER_APP.value,
            "source_turn": 4,
            "evidence": "We'll decide fees later.",
            "decision_summary": "Fee policy",
            "reopened_at_turn": 9,
            "explicit_authorization": True,
        }
    ]

    result = pm.pm_compile_node(initial)

    assert result["pm_is_complete"] is True
    assert result["prd_contract"].deferred_decisions == []


def test_constraint_source_is_not_required_as_visible_citation():
    primary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="user",
        evidence="A user uses the app",
        roles=["user"],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    no_secondary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="secondary_users",
        value="none",
        evidence="There are no other users",
        roles=[],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
        absence="none",
    )
    sources = build_source_snapshot(state(primary, no_secondary))
    projection = project_prd(sources)

    # Structural coverage is satisfied even though the absence source is not
    # rendered as a persona/scope/requirement.
    validate_projection(projection, sources)


def test_constraint_source_cannot_be_forced_into_visible_claims():
    primary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="user",
        evidence="A user uses the app",
        roles=["user"],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    no_secondary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="secondary_users",
        value="none",
        evidence="There are no other users",
        roles=[],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
        absence="none",
    )
    sources = build_source_snapshot(state(primary, no_secondary))
    projection = project_prd(sources)
    absent = next(source for source in sources if source.key == "secondary_users")

    projection.draft.scope.in_scope.append(
        SourcedClaim(
            text="There are no secondary users.",
            category="USER_ROLES.secondary_users",
            actor_ids=[],
            conditions=[],
            source_fact_ids=[absent.fact_id],
        )
    )

    with pytest.raises(ValueError, match="Constraint/unknown sources leaked"):
        validate_projection(projection, sources)


def test_all_factual_sections_are_enumerated_for_semantic_validation():
    source = build_source_snapshot(state())[0]
    ref = dict(
        source_fact_ids=[source.fact_id],
        category="BUSINESS_RULES.approval_rules",
        actor_ids=["manager"],
        conditions=["over $100"],
    )
    claim = dict(**ref, text=QUOTE)
    value = dict(
        product_name=claim,
        elevator_pitch=[claim],
        scope=dict(in_scope=[claim], out_of_scope=[claim]),
        personas=[
            dict(
                **ref,
                name="Manager",
                description=QUOTE,
                key_behaviors=[claim],
            )
        ],
        functional_requirements=[
            dict(
                **ref,
                id="FR-01",
                description=QUOTE,
                validation="TBD",
            )
        ],
        non_functional_constraints=[claim],
        deferred_items=[claim],
        open_questions=[],
    )

    assert len(list(draft_claims(PRDDraft.model_validate(value)))) == 9


def test_graph_waits_for_founder_confirmation_before_compilation(monkeypatch):
    from agents import graph

    called = {"compile": 0}

    monkeypatch.setattr(graph, "knowledge_tracker_node", lambda _: {})
    monkeypatch.setattr(
        graph,
        "interview_planner_node",
        lambda _: {
            "awaiting_confirmation": False,
            "prd_confirmation_pending": True,
            "ready_to_compile": False,
        },
    )
    monkeypatch.setattr(graph, "all_discovery_resolved", lambda _: True)
    monkeypatch.setattr(
        graph,
        "pm_compile_node",
        lambda _: called.__setitem__("compile", called["compile"] + 1) or {},
    )

    initial = state()
    initial.update(
        {
            "messages": [HumanMessage(content="idea")],
            "conversation_intent": "product_information",
            "checkpoint_cursor": "waiting",
        }
    )
    result = graph.build_graph().invoke(initial)

    assert called["compile"] == 0
    assert result["prd_confirmation_pending"] is True
    assert result["ready_to_compile"] is False
    assert result["messages"][-1].content == graph.PRD_CONFIRMATION_PROMPT



def test_projected_persona_includes_grounded_actor_responsibilities():
    primary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="user",
        evidence="A user manages tasks",
        roles=["user"],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    action = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="responsibilities",
        value="create tasks",
        evidence="A user can create tasks",
        role="user",
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )

    sources = build_source_snapshot(state(primary, action))
    projection = project_prd(sources)
    validate_projection(projection, sources)

    persona = projection.draft.personas[0]
    assert persona.name == "User"
    assert [behavior.text for behavior in persona.key_behaviors] == [
        "create tasks"
    ]
    assert projection.draft.functional_requirements[0].description == (
        "create tasks"
    )


def test_persona_behavior_prose_edits_cannot_change_provenance():
    primary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="user",
        evidence="A user manages tasks",
        roles=["user"],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    action = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="responsibilities",
        value="create tasks",
        evidence="A user can create tasks",
        role="user",
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )

    sources = build_source_snapshot(state(primary, action))
    projection = project_prd(sources)
    original = projection.draft.personas[0].key_behaviors[0]

    edited = apply_prose_edits(
        projection.draft,
        [
            PRDProseEdit(
                claim_id="personas/0/key_behaviors/0",
                text="Creates personal tasks.",
            )
        ],
    )
    changed = edited.personas[0].key_behaviors[0]

    assert changed.text == "Creates personal tasks."
    assert changed.category == original.category
    assert changed.actor_ids == original.actor_ids
    assert changed.source_fact_ids == original.source_fact_ids



def test_todo_secondary_user_absence_cannot_block_prd_save(monkeypatch, tmp_path):
    primary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="user",
        evidence="A user can create tasks, edit or delete them, and mark them as completed",
        roles=["user"],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    no_secondary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="secondary_users",
        value="none",
        evidence="There are no other user roles, payments, integrations, or admin features",
        roles=[],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
        absence="none",
    )
    actions = [
        KnowledgeItem(
            topic=T.USER_ROLES,
            scope=S.USER_APP,
            key="responsibilities",
            value=value,
            evidence="A user can create tasks, edit or delete them, and mark them as completed",
            role="user",
            confidence=1,
            knowledge_state=K.CONFIRMED,
            source_turn=0,
        )
        for value in (
            "create tasks",
            "edit tasks",
            "delete tasks",
            "mark tasks as completed",
        )
    ]
    exclusions = [
        KnowledgeItem(
            topic=T.MVP_SCOPE,
            scope=S.USER_APP,
            key="out_of_scope",
            value=value,
            evidence="There are no other user roles, payments, integrations, or admin features",
            confidence=1,
            knowledge_state=K.CONFIRMED,
            source_turn=0,
        )
        for value in ("payments", "integrations", "admin features")
    ]
    concept = ProductConcept(
        kind=ProductConceptKind.ENTITY,
        scope=S.USER_APP,
        subject="task",
        value="Tasks can be either active or completed",
        evidence="Tasks can be either active or completed",
        confidence=1,
        source_turn=0,
    )
    initial = state(primary, no_secondary, *actions, *exclusions)
    initial["product_concepts"] = [concept.model_dump(mode="json")]

    install_prose(monkeypatch, parsed={"edits": []})
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)

    result = pm.pm_compile_node(initial)

    assert result["pm_is_complete"] is True
    assert result["compilation_errors"] == []
    contract = result["prd_contract"]
    absent_source = next(
        source for source in contract.source_facts
        if source.key == "secondary_users"
    )
    assert absent_source.fact_id in contract.constraint_source_ids
    assert [persona.name for persona in contract.personas] == ["User"]
    assert {
        requirement.description
        for requirement in contract.functional_requirements
    } >= {
        "create tasks",
        "edit tasks",
        "delete tasks",
        "mark tasks as completed",
        "Tasks can be either active or completed",
    }
    assert {item.text for item in contract.scope.out_of_scope} == {
        "payments",
        "integrations",
        "admin features",
    }
    assert (tmp_path / "requirements_mvp.json").exists()



def test_todo_product_model_groups_atomic_facts_into_feature_specifications():
    primary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="user",
        evidence="A user manages their own tasks",
        roles=["user"],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    actions = [
        KnowledgeItem(
            topic=T.USER_ROLES,
            scope=S.USER_APP,
            key="responsibilities",
            value=value,
            evidence="A user can create tasks, edit or delete them, and mark them as completed",
            role="user",
            confidence=1,
            knowledge_state=K.CONFIRMED,
            source_turn=0,
        )
        for value in (
            "create tasks",
            "edit tasks",
            "delete tasks",
            "mark tasks as completed",
        )
    ]
    persistence = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="responsibilities",
        value="access their tasks across devices",
        evidence="The user should be able to access their tasks across devices",
        role="user",
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=2,
    )
    concepts = [
        ProductConcept(
            kind=ProductConceptKind.ENTITY,
            scope=S.USER_APP,
            subject="task",
            value="Tasks can be either active or completed",
            evidence="Tasks can be either active or completed",
            confidence=1,
            source_turn=0,
        ),
        ProductConcept(
            kind=ProductConceptKind.ATTRIBUTE,
            scope=S.USER_APP,
            subject="task",
            relation="has attribute",
            object="required title",
            value="Each task should have a title",
            evidence="Each task should have a title",
            confidence=1,
            source_turn=1,
        ),
        ProductConcept(
            kind=ProductConceptKind.ATTRIBUTE,
            scope=S.USER_APP,
            subject="task",
            relation="has attribute",
            object="optional description",
            value="an optional description",
            evidence="an optional description",
            source_question="What information should each task contain?",
            confidence=1,
            source_turn=1,
        ),
        ProductConcept(
            kind=ProductConceptKind.ATTRIBUTE,
            scope=S.USER_APP,
            subject="task",
            relation="has attribute",
            object="due date",
            value="a due date",
            evidence="a due date",
            source_question="What information should each task contain?",
            confidence=1,
            source_turn=1,
        ),
        ProductConcept(
            kind=ProductConceptKind.ATTRIBUTE,
            scope=S.USER_APP,
            subject="task",
            relation="has attribute",
            object="priority level",
            value="a priority level",
            evidence="a priority level",
            source_question="What information should each task contain?",
            confidence=1,
            source_turn=1,
        ),
        ProductConcept(
            kind=ProductConceptKind.ATTRIBUTE,
            scope=S.USER_APP,
            subject="task",
            relation="tracks",
            object="active or completed status",
            value="It should also track whether the task is active or completed",
            evidence="It should also track whether the task is active or completed",
            confidence=1,
            source_turn=1,
        ),
        ProductConcept(
            kind=ProductConceptKind.RELATIONSHIP,
            scope=S.USER_APP,
            subject="task",
            relation="belongs to",
            object="user account",
            value="Tasks should be tied to a user account and stored in the cloud",
            evidence="Tasks should be tied to a user account and stored in the cloud",
            confidence=1,
            source_turn=2,
        ),
    ]
    initial = state(primary, *actions, persistence)
    initial["product_concepts"] = [
        concept.model_dump(mode="json") for concept in concepts
    ]

    sources = build_source_snapshot(initial)
    projection = project_prd(sources)
    model = build_product_model(sources)
    features = build_feature_specifications(projection.draft, model)
    validate_feature_specifications(features, projection.draft, model)

    assert [feature.title for feature in features] == [
        "Task Management",
        "Task Persistence & Access",
    ]

    task_management = features[0]
    assert len(task_management.details) == 3
    responsibility = next(
        detail
        for detail in task_management.details
        if detail.category == "USER_ROLES.responsibilities"
    )
    attributes = next(
        detail
        for detail in task_management.details
        if detail.category == "PRODUCT_MODEL.attribute"
    )
    assert responsibility.text == (
        "User can create tasks, edit tasks, delete tasks, and mark tasks as completed."
    )
    assert attributes.text == (
        "Task details: required title, optional description, due date, priority level, "
        "and active or completed status."
    )

    persistence_feature = features[1]
    assert {
        detail.category for detail in persistence_feature.details
    } == {
        "USER_ROLES.responsibilities",
        "PRODUCT_MODEL.relationship",
    }


def test_compiled_todo_contract_contains_grouped_feature_specifications(
    monkeypatch,
    tmp_path,
):
    primary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="user",
        evidence="A user manages tasks",
        roles=["user"],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    actions = [
        KnowledgeItem(
            topic=T.USER_ROLES,
            scope=S.USER_APP,
            key="responsibilities",
            value=value,
            evidence="A user can create tasks, edit or delete them, and mark them as completed",
            role="user",
            confidence=1,
            knowledge_state=K.CONFIRMED,
            source_turn=0,
        )
        for value in ("create tasks", "edit tasks", "delete tasks", "mark tasks as completed")
    ]
    concept = ProductConcept(
        kind=ProductConceptKind.ATTRIBUTE,
        scope=S.USER_APP,
        subject="task",
        relation="has attribute",
        object="required title",
        value="Each task should have a title",
        evidence="Each task should have a title",
        confidence=1,
        source_turn=1,
    )
    initial = state(primary, *actions)
    initial["product_concepts"] = [concept.model_dump(mode="json")]
    install_prose(monkeypatch, parsed={"edits": []})
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)

    result = pm.pm_compile_node(initial)

    assert result["pm_is_complete"] is True
    contract = result["prd_contract"]
    assert contract.schema_version == "2.1"
    assert [feature.title for feature in contract.feature_specifications] == [
        "Task Management"
    ]
    assert contract.feature_specifications[0].details[0].text.startswith("User can ")
    saved = json.loads((tmp_path / "requirements_mvp.json").read_text())
    assert saved["feature_specifications"][0]["title"] == "Task Management"



def test_feature_grouping_uses_atomic_meaning_not_shared_evidence_text():
    shared = (
        "A user can create tasks, and tasks are stored in the cloud for access "
        "across devices"
    )
    action = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="responsibilities",
        value="create tasks",
        evidence=shared,
        role="user",
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    persistence = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="responsibilities",
        value="access tasks across devices",
        evidence=shared,
        role="user",
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    entity = ProductConcept(
        kind=ProductConceptKind.ENTITY,
        scope=S.USER_APP,
        subject="task",
        value="Tasks are personal task records",
        evidence=shared,
        confidence=1,
        source_turn=0,
    )
    initial = state(action, persistence)
    initial["product_concepts"] = [entity.model_dump(mode="json")]

    sources = build_source_snapshot(initial)
    projection = project_prd(sources)
    model = build_product_model(sources)
    features = build_feature_specifications(projection.draft, model)

    by_title = {feature.title: feature for feature in features}
    assert "Task Management" in by_title
    assert "Task Persistence & Access" in by_title
    management_values = {
        ref
        for detail in by_title["Task Management"].details
        for ref in detail.source_fact_ids
    }
    action_source = next(
        source for source in sources
        if source.value == "create tasks"
    )
    persistence_source = next(
        source for source in sources
        if source.value == "access tasks across devices"
    )
    assert action_source.fact_id in management_values
    assert persistence_source.fact_id not in management_values
