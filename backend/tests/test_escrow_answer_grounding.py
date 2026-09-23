"""Captured real extraction output must survive unrelated grounding failures."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from agents import graph, guardrails, knowledge_tracker as tracker, question_generator
from agents.semantic_validation import GapAnswer, GroundingResult
from agents.state import DiscoveryScope as S, KnowledgeState as K
from replay_escrow_responsibilities import initial_state, ANSWER, ACTORS


CAPTURE = json.loads((Path(__file__).parent / "fixtures" / "escrow_responsibilities_extraction.json").read_text(encoding="utf-8-sig"))


def setup_models(monkeypatch):
    prompts = {}
    def invoke(name, messages):
        prompts[name] = messages[0].content
        return {"parsed": {"items": CAPTURE.get(name, [])}}
    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, n=name: invoke(n, messages))
        for name, *_ in tracker.PASSES})
    return prompts


@pytest.mark.parametrize("incidental_failure", ["malformed", "rejected"])
def test_exact_answer_advances_even_when_incidental_audit_fails(monkeypatch, incidental_failure):
    prompts = setup_models(monkeypatch)
    audits = []
    def decide(name, schema, instruction, payload):
        if name == "GAP_ANSWER":
            return GapAnswer(resolution="unresolved", evidence=ANSWER, confidence=1)
        audits.append(payload)
        candidates = payload["candidates"]
        if all(c["key"] == "responsibilities" for c in candidates):
            assert len(candidates) == 2
            assert {c["role"] for c in candidates} == {"customer"}
            assert any(a["evidence"] == ACTORS and a["roles"] == ["customer"]
                       for a in payload["confirmed_actor_context"])
            return GroundingResult(supported_ids=[c["id"] for c in candidates],
                evidence_categories={str(c["evidence_id"]): ["USER_ROLES.responsibilities"] for c in candidates})
        if incidental_failure == "malformed":
            raise ValueError("Duplicate evidence category verdicts")
        return GroundingResult(supported_ids=[], evidence_categories={},
                               rejection_reasons={str(c["id"]): "unsupported" for c in candidates})
    monkeypatch.setattr(tracker, "semantic_decision", decide)
    monkeypatch.setattr(question_generator, "ChatOllama", lambda **_: SimpleNamespace(invoke=lambda _:
        AIMessage(content="Are there actions a customer must not be allowed to perform?")))
    monkeypatch.setattr(guardrails, "evaluator_llm", SimpleNamespace(invoke=lambda _: SimpleNamespace(passed=True)))
    result = graph.build_graph().invoke(initial_state())
    assert len(audits) == 2
    assert result["current_gap"] == "permissions::customer"
    assert result["question_retry_count"] == 0
    duties = [i for i in result["discovered_knowledge"] if i.key == "responsibilities"]
    assert len(duties) == 2 and all(i.role == "customer" for i in duties)
    assert any("buyer" in i.evidence for i in duties)
    assert any(i.evidence.startswith("The seller") for i in duties)
    assert ACTORS in prompts["RESPONSIBILITY"]
    assert not any("trouble" in str(m.content) for m in result["messages"])


def test_rejected_active_answer_is_not_promoted_just_to_advance(monkeypatch):
    setup_models(monkeypatch)
    monkeypatch.setattr(tracker, "semantic_decision", lambda name, *args:
        GapAnswer(resolution="unresolved", evidence=ANSWER, confidence=1) if name == "GAP_ANSWER"
        else GroundingResult(supported_ids=[], evidence_categories={}))
    result = tracker.knowledge_tracker_node(initial_state())
    assert not any(i.key == "responsibilities" for i in result["discovered_knowledge"])


def test_identity_context_excludes_other_scope_and_unconfirmed_relationships():
    state = initial_state()
    actor = state["discovered_knowledge"][0]
    state["discovered_knowledge"] += [
        actor.model_copy(update={"scope": S.ADMIN_DASHBOARD, "value": "unrelated admin"}),
        actor.model_copy(update={"knowledge_state": K.INFERRED, "value": "unconfirmed relationship"})]
    context = tracker.confirmed_actor_context(state, S.USER_APP)
    assert len(context) == 1 and context[0]["evidence"] == ACTORS
