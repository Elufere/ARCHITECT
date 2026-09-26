from adaptive_pm.engine import AdaptivePMEngine
from adaptive_pm.models import (
    CapturedFact,
    DecisionRecord,
    InterviewState,
    KnowledgeMutation,
    KnowledgeRecord,
    KnowledgeStatus,
    Observation,
    PlanningResult,
    CompletionAssessment,
    QuestionCandidate,
    ReasoningUpdate,
    TurnCapture,
)


def engine_without_models():
    return AdaptivePMEngine.__new__(AdaptivePMEngine)


def incomplete():
    return CompletionAssessment(
        complete=False,
        core_product_model_coherent=False,
        major_entities_understood=False,
        primary_workflows_understood=False,
        high_impact_rules_understood=False,
        architecture_changing_unknowns_resolved_or_deferred=False,
        transactional_mechanics_understood=False,
        lifecycle_understood=False,
        major_failure_paths_addressed=False,
        mvp_boundaries_clear=False,
        reason="still discovering",
    )


def candidate(identity: str, decision_key: str) -> QuestionCandidate:
    return QuestionCandidate(
        id=identity,
        decision_key=decision_key,
        question="What completes the transaction?",
        uncertainty="Completion rule is unknown",
        why_now="It defines the core lifecycle",
        business_impact=1,
        architecture_impact=1,
        dependency_unlock=1,
        uncertainty_reduction=1,
        risk_reduction=1,
        contextual_relevance=1,
    )


def test_source_grounding_drops_invented_evidence():
    capture = TurnCapture(facts=[
        CapturedFact(
            id="fact_1",
            statement="The host requires approval.",
            evidence="requires approval",
        )
    ])
    grounded = engine_without_models()._source_ground(
        capture,
        "The host can create an event.",
    )
    assert grounded.facts == []


def test_confirmed_knowledge_requires_grounded_observation():
    state = InterviewState()
    update = ReasoningUpdate(knowledge_mutations=[
        KnowledgeMutation(
            action="ADD",
            key="payments.provider",
            statement="Flutterwave is the payment provider.",
            category="payments",
            status=KnowledgeStatus.CONFIRMED,
            observation_ids=["missing"],
        )
    ])
    AdaptivePMEngine._apply_reasoning(state, update)
    assert "payments.provider" not in state.knowledge


def test_refinement_preserves_prior_evidence():
    state = InterviewState(turn_count=2)
    first = Observation(
        id="obs_1",
        statement="Guests can pay.",
        evidence="Guests can pay",
        source_turn=1,
    )
    second = Observation(
        id="obs_2",
        statement="Guests pay with Flutterwave.",
        evidence="Guests pay with Flutterwave",
        source_turn=2,
    )
    state.observations = [first, second]
    state.knowledge["payments.method"] = KnowledgeRecord(
        key="payments.method",
        statement="Guests can pay.",
        category="payments",
        evidence_observation_ids=["obs_1"],
        first_seen_turn=1,
        last_updated_turn=1,
    )
    update = ReasoningUpdate(knowledge_mutations=[
        KnowledgeMutation(
            action="REFINE",
            key="payments.method",
            statement="Guests pay with Flutterwave.",
            category="payments",
            observation_ids=["obs_2"],
        )
    ])
    AdaptivePMEngine._apply_reasoning(state, update)
    assert state.knowledge["payments.method"].evidence_observation_ids == [
        "obs_1",
        "obs_2",
    ]


def test_answered_decision_is_not_eligible_again():
    state = InterviewState()
    state.decisions["transaction.completion"] = DecisionRecord(
        decision_key="transaction.completion",
        question="What completes the transaction?",
        asked_turn=4,
        resolution="ANSWERED",
    )
    plan = PlanningResult(
        completion=incomplete(),
        candidates=[
            candidate("q1", "transaction.completion"),
            candidate("q2", "settlement.destination"),
        ],
    )
    eligible = AdaptivePMEngine._eligible_candidates(plan, state)
    assert [item.decision_key for item in eligible] == ["settlement.destination"]


def test_decision_memory_is_not_a_short_sliding_window():
    state = InterviewState()
    for index in range(20):
        key = f"decision.{index}"
        state.decisions[key] = DecisionRecord(
            decision_key=key,
            question=f"Question {index}?",
            asked_turn=index,
            resolution="ANSWERED",
        )
    assert len(state.compact_context()["decisions"]) == 20
