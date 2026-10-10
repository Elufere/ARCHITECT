import json

from agents.prd_schema import (
    ClaimVerdict,
    PRDContract,
    PRDDraft,
    ScopeBoundary,
    SourceFact,
    SourcedClaim,
    UserPersona,
)
from agents.prd_validation import validate_prd
from services.workspace import _prd_sections


def _source(
    fact_id: str,
    topic: str,
    key: str,
    value: str,
    *,
    evidence: str | None = None,
    role: str | None = None,
    roles: list[str] | None = None,
    subject: str | None = None,
    relation: str | None = None,
    object: str | None = None,
    source_question: str | None = None,
) -> SourceFact:
    return SourceFact(
        fact_id=fact_id,
        topic=topic,
        scope="USER_APP",
        key=key,
        value=value,
        evidence=evidence or value,
        source_question=source_question,
        roles=roles,
        aliases=None,
        role=role,
        confidence=1.0,
        knowledge_state="CONFIRMED",
        source_turn=0,
        absence=None,
        subject=subject,
        relation=relation,
        object=object,
    )


class _Classifier:
    def invoke(self, messages):
        payload = json.loads(messages[-1].content)
        if "evidence" in payload:
            # Simulate the production failure: a shared founder sentence contains
            # several true assertions and the blind classifier notices a sibling
            # category rather than the narrower atomic source category.
            return {
                "categories": ["USER_ROLES.secondary_users"],
                "explanation": "The sentence also discusses user roles.",
            }
        return {
            "categories": ["MVP_SCOPE.out_of_scope"],
            "explanation": "The claim explicitly excludes a capability.",
        }


class _Auditor:
    def invoke(self, messages):
        payload = json.loads(messages[-1].content)
        return ClaimVerdict(
            claim_id=payload["claim_id"],
            source_evidence_supports_facts=True,
            claim_supported=True,
            category_preserved=True,
            actors_preserved=True,
            conditions_preserved=True,
            validation_supported=True,
            no_conflict_with_confirmed_facts=True,
            explanation="Supported.",
        )


def test_shared_evidence_does_not_discard_safe_prd_wording():
    source = _source(
        "fact_payments",
        "MVP_SCOPE",
        "out_of_scope",
        "payments",
        evidence=(
            "There are no other user roles, payments, integrations, "
            "notifications, or admin features."
        ),
    )
    draft = PRDDraft(
        product_name=None,
        elevator_pitch=[],
        scope=ScopeBoundary(
            in_scope=[],
            out_of_scope=[
                SourcedClaim(
                    text="Payments are out of scope.",
                    category="MVP_SCOPE.out_of_scope",
                    actor_ids=[],
                    conditions=[],
                    source_fact_ids=[source.fact_id],
                )
            ],
        ),
        personas=[],
        functional_requirements=[],
        non_functional_constraints=[],
        deferred_items=[],
        open_questions=[],
    )

    verdicts = validate_prd(
        draft,
        [source],
        _Auditor(),
        _Classifier(),
        {},
    )

    assert len(verdicts) == 1



def test_semantic_audit_can_be_scoped_to_only_prose_edits():
    edited = _source(
        "fact_payments",
        "MVP_SCOPE",
        "out_of_scope",
        "payments",
        evidence="There are no payments.",
    )
    unchanged = _source(
        "fact_action",
        "USER_ROLES",
        "responsibilities",
        "create tasks",
        evidence="A user can create tasks.",
        role="user",
    )
    draft = PRDDraft(
        product_name=None,
        elevator_pitch=[],
        scope=ScopeBoundary(
            in_scope=[],
            out_of_scope=[
                SourcedClaim(
                    text="Payments are out of scope.",
                    category="MVP_SCOPE.out_of_scope",
                    actor_ids=[],
                    conditions=[],
                    source_fact_ids=[edited.fact_id],
                )
            ],
        ),
        personas=[],
        functional_requirements=[
            {
                "id": "FR-01",
                "description": "create tasks",
                "validation": "TBD",
                "category": "USER_ROLES.responsibilities",
                "actor_ids": ["user"],
                "conditions": [],
                "source_fact_ids": [unchanged.fact_id],
            }
        ],
        non_functional_constraints=[],
        deferred_items=[],
        open_questions=[],
    )

    class EditedOnlyClassifier:
        def invoke(self, messages):
            payload = json.loads(messages[-1].content)
            text = json.dumps(payload).lower()
            if "payments are out of scope" in text:
                return {
                    "categories": ["MVP_SCOPE.out_of_scope"],
                    "explanation": "Explicit scope exclusion.",
                }
            raise AssertionError("Unedited deterministic claim should not be classified")

    verdicts = validate_prd(
        draft,
        [edited, unchanged],
        _Auditor(),
        EditedOnlyClassifier(),
        {},
        claim_ids={"scope/out_of_scope/0"},
    )

    assert len(verdicts) == 1
    assert verdicts[0].claim_id == "scope/out_of_scope/0"

def test_todo_prd_rendering_is_semantic_not_storage_shaped():
    sources = [
        _source(
            "fact_goal",
            "USER_GOALS",
            "primary_user_goals",
            "the tasks should be available across all devices",
            role="user",
        ),
        _source(
            "fact_permission",
            "USER_ROLES",
            "permissions",
            "users should sign in before using the app",
            role="user",
        ),
        _source(
            "fact_create",
            "USER_ROLES",
            "responsibilities",
            "create tasks",
            role="user",
        ),
        _source(
            "fact_edit",
            "USER_ROLES",
            "responsibilities",
            "edit tasks",
            role="user",
        ),
        _source(
            "fact_delete",
            "USER_ROLES",
            "responsibilities",
            "delete tasks whether active or completed",
            role="user",
        ),
        _source(
            "fact_reopen",
            "USER_ROLES",
            "responsibilities",
            "move completed tasks back to active",
            role="user",
        ),
        _source(
            "fact_deleted",
            "CORE_WORKFLOW",
            "end_state",
            "deleted tasks are removed from the app immediately",
        ),
        _source(
            "fact_initial",
            "BUSINESS_RULES",
            "validation_rules",
            "new tasks can only start as active",
        ),
        _source(
            "concept_task",
            "PRODUCT_MODEL",
            "entity",
            "Tasks can be either active or completed",
            subject="task",
        ),
        _source(
            "concept_title",
            "PRODUCT_MODEL",
            "attribute",
            "Each task has a title",
            subject="task",
            relation="has",
            object="title",
        ),
        _source(
            "concept_description",
            "PRODUCT_MODEL",
            "attribute",
            "optional description",
            subject="task",
            relation="has",
            object="description",
        ),
    ]

    contract = PRDContract.model_validate(
        {
            "schema_version": "2.1",
            "discovery_scope": "USER_APP",
            "product_name": None,
            "elevator_pitch": [
                {
                    "text": "the tasks should be available across all devices",
                    "category": "USER_GOALS.primary_user_goals",
                    "actor_ids": ["user"],
                    "conditions": [],
                    "source_fact_ids": ["fact_goal"],
                }
            ],
            "scope": {"in_scope": [], "out_of_scope": []},
            "personas": [
                {
                    "name": "User",
                    "description": "Primary product user.",
                    "key_behaviors": [
                        {
                            "text": "delete tasks whether active or completed",
                            "category": "USER_ROLES.responsibilities",
                            "actor_ids": ["user"],
                            "conditions": [],
                            "source_fact_ids": ["fact_delete"],
                        }
                    ],
                    "category": "USER_ROLES.primary_users",
                    "actor_ids": ["user"],
                    "conditions": [],
                    "source_fact_ids": ["fact_create"],
                }
            ],
            "functional_requirements": [],
            "non_functional_constraints": [],
            "deferred_items": [],
            "open_questions": [],
            "source_facts": [source.model_dump(mode="json") for source in sources],
            "validation_report": [],
            "external_systems": [],
            "deferred_decisions": [
                {
                    "id": "deferral:persistence",
                    "kind": "design_implementation",
                    "decision": (
                        "Clarify how and where task data is stored and persisted "
                        "for cross-device access."
                    ),
                    "evidence": "Leave this to design or engineering.",
                    "source_turn": 3,
                    "resolution_stage": None,
                    "owner": None,
                    "downstream_consequence": None,
                    "requirement_id": None,
                }
            ],
            "feature_specifications": [
                {
                    "id": "task-management",
                    "title": "Task Management",
                    "overview": "Users manage task records and their lifecycle.",
                    "details": [
                        {
                            "text": "Users can create tasks and edit tasks.",
                            "category": "USER_ROLES.responsibilities",
                            "actor_ids": ["user"],
                            "conditions": [],
                            "source_fact_ids": ["fact_create", "fact_edit"],
                        },
                        {
                            "text": (
                                "Users can delete tasks whether active or completed "
                                "and move completed tasks back to active."
                            ),
                            "category": "USER_ROLES.responsibilities",
                            "actor_ids": ["user"],
                            "conditions": [],
                            "source_fact_ids": ["fact_delete", "fact_reopen"],
                        },
                        {
                            "text": "Users should sign in before using the app.",
                            "category": "USER_ROLES.permissions",
                            "actor_ids": ["user"],
                            "conditions": [],
                            "source_fact_ids": ["fact_permission"],
                        },
                    ],
                    "requirement_ids": [],
                    "source_fact_ids": [
                        "fact_create", "fact_edit", "fact_delete",
                        "fact_reopen", "fact_permission",
                    ],
                }
            ],
            "constraint_source_ids": [],
            "prose_polished": False,
        }
    )

    sections = _prd_sections(
        contract,
        (
            "Simple To-Do is a personal task management app where a user can "
            "create, edit, delete, and complete tasks."
        ),
    )
    by_id = {section.id: section for section in sections}

    assert "the tasks should be available across all devices" not in by_id["overview"].body
    assert "the tasks should be available across all devices" in by_id["goals"].body

    assert by_id["users"].body == "User — Primary product user."
    assert "delete tasks" not in by_id["users"].body

    assert by_id["user_flow"].title == "Core User Flow & Lifecycle"
    assert "Access / permissions:" not in by_id["user_flow"].body
    assert "users should sign in before using the app" not in by_id["user_flow"].body
    assert "User actions:" not in by_id["user_flow"].body
    assert "move completed tasks back to active" not in by_id["user_flow"].body
    assert "users should sign in before using the app" in by_id["feature_1"].body
    assert "move completed tasks back to active" in by_id["feature_1"].body
    assert "Lifecycle / business rules:" in by_id["user_flow"].body
    assert "new tasks can only start as active" in by_id["user_flow"].body
    assert "Resulting state:" in by_id["user_flow"].body

    assert "States: active, completed" in by_id["data_model"].body
    assert "- title" in by_id["data_model"].body
    assert "- description (optional)" in by_id["data_model"].body
    assert "Has:" not in by_id["data_model"].body

    assert "deferred" not in by_id
    assert "implementation_decisions" in by_id
    assert by_id["implementation_decisions"].title == "Implementation Decisions"
    assert "Delegated to design / engineering:" in by_id["implementation_decisions"].body
