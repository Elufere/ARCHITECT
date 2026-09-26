from pm_v2.engine import PMDiscoveryEngine
from pm_v2.models import DiscoveryState, QuestionCandidate
from pm_v2.provenance import locate_exact_evidence, provenance_valid


def test_provenance_requires_exact_current_turn_quote():
    source = "Guests can buy packages without an account."
    assert provenance_valid(source, "Guests can buy packages")
    assert not provenance_valid(source, "Hosts can buy packages")


def test_provenance_returns_offsets():
    source = "abc guest checkout xyz"
    span = locate_exact_evidence(source, "guest checkout")
    assert span is not None
    assert source[span.start:span.end] == "guest checkout"


def test_question_priority_penalizes_repetition_and_premature_detail():
    useful = QuestionCandidate(
        decision_key="payment.release_condition",
        objective="Determine what completes payment release.",
        question="What has to happen before the seller receives the money?",
        business_impact=1.0,
        architecture_impact=0.9,
        dependency_unlock=1.0,
        uncertainty=1.0,
        risk=1.0,
        contextual_relevance=1.0,
    )
    weak = QuestionCandidate(
        decision_key="checkout.button_style",
        objective="Determine a cosmetic detail.",
        question="How should the checkout button look?",
        business_impact=0.1,
        architecture_impact=0.0,
        dependency_unlock=0.0,
        uncertainty=0.2,
        risk=0.0,
        contextual_relevance=0.2,
        repetition_penalty=0.4,
        premature_detail_penalty=1.0,
        user_fatigue_penalty=0.5,
    )
    assert PMDiscoveryEngine._priority_score(useful) > PMDiscoveryEngine._priority_score(weak)


def test_blocked_decision_keys_survive_as_state_not_transcript_memory():
    state = DiscoveryState(session_id="00000000-0000-0000-0000-000000000001")
    assert PMDiscoveryEngine._blocked_decision_keys(state) == set()
