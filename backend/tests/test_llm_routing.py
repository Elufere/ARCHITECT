from agents import llm


def test_expensive_reasoning_is_routed_only_to_selected_calls(monkeypatch):
    monkeypatch.delenv("OPENAI_FORCE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_CALLS", raising=False)
    monkeypatch.setenv("OPENAI_FAST_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_REASONING_MODEL", "gpt-5.6-sol")

    assert llm._model_for_call("discovery_threads.plan") == "gpt-5.6-sol"
    assert llm._model_for_call("pm_compile.prose") == "gpt-5.6-sol"

    for call_name in (
        "knowledge_tracker.CLAIMS",
        "knowledge_tracker.GROUNDING",
        "discovery_threads.assess_inquiry",
        "question_generator",
        "guardrail",
        "pm_compile.audit",
        "pm_compile.classification",
    ):
        assert llm._model_for_call(call_name) == "gpt-4.1-mini"


def test_legacy_openai_model_does_not_accidentally_force_sol(monkeypatch):
    monkeypatch.delenv("OPENAI_FORCE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_CALLS", raising=False)
    monkeypatch.delenv("OPENAI_FAST_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")

    assert llm._model_for_call("knowledge_tracker.CLAIMS") == "gpt-4.1-mini"
    assert llm._model_for_call("discovery_threads.plan") == "gpt-5.6-sol"


def test_force_model_is_explicit_global_override(monkeypatch):
    monkeypatch.setenv("OPENAI_FORCE_MODEL", "gpt-4.1-mini")

    assert llm._model_for_call("discovery_threads.plan") == "gpt-4.1-mini"
    assert llm._model_for_call("pm_compile.prose") == "gpt-4.1-mini"


def test_reasoning_calls_are_configurable(monkeypatch):
    monkeypatch.delenv("OPENAI_FORCE_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_REASONING_CALLS", "discovery_threads.plan")
    monkeypatch.setenv("OPENAI_REASONING_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("OPENAI_FAST_MODEL", "gpt-4.1-mini")

    assert llm._model_for_call("discovery_threads.plan") == "gpt-5.6-sol"
    assert llm._model_for_call("pm_compile.prose") == "gpt-4.1-mini"
