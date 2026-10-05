from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import services.prd_generation as prd_generation


PROJECT_ID = "escrow-app-a1b2c3d4"
SESSION_ID = "session-123"


def _project():
    return SimpleNamespace(
        id=PROJECT_ID,
        discovery_session_id=SESSION_ID,
    )


def _approved_state(**updates):
    state = {
        "session_id": SESSION_ID,
        "prd_contract": None,
        "prd_confirmation_pending": False,
        "ready_to_compile": True,
        "checkpoint_cursor": "waiting",
        "interview_status": "WAITING_FOR_USER",
        "compilation_errors": [],
    }
    state.update(updates)
    return state


def _patch_project_boundary(monkeypatch):
    monkeypatch.setattr(prd_generation, "get_project", lambda project_id: _project())
    monkeypatch.setattr(
        prd_generation,
        "session_lock",
        lambda session_id: nullcontext(),
    )
    monkeypatch.setattr(
        prd_generation,
        "build_workspace_snapshot",
        lambda project_id: {"project_id": project_id, "prd": "ready"},
    )
    monkeypatch.setattr(
        prd_generation,
        "all_discovery_resolved",
        lambda state: True,
    )


def test_generate_prd_requires_explicit_founder_approval(monkeypatch):
    _patch_project_boundary(monkeypatch)
    monkeypatch.setattr(
        prd_generation,
        "load_checkpoint",
        lambda session_id: _approved_state(ready_to_compile=False),
    )

    with pytest.raises(
        prd_generation.PrdGenerationStateError,
        match="not been authorized",
    ):
        prd_generation.generate_project_prd(PROJECT_ID)


def test_generate_prd_rejects_pending_confirmation(monkeypatch):
    _patch_project_boundary(monkeypatch)
    monkeypatch.setattr(
        prd_generation,
        "load_checkpoint",
        lambda session_id: _approved_state(
            prd_confirmation_pending=True,
            ready_to_compile=False,
        ),
    )

    with pytest.raises(
        prd_generation.PrdGenerationStateError,
        match="founder confirmation",
    ):
        prd_generation.generate_project_prd(PROJECT_ID)


def test_generate_prd_is_idempotent_after_verified_contract(monkeypatch):
    _patch_project_boundary(monkeypatch)
    monkeypatch.setattr(
        prd_generation,
        "load_checkpoint",
        lambda session_id: _approved_state(prd_contract=object()),
    )

    class UnexpectedGraph:
        def invoke(self, *args, **kwargs):
            raise AssertionError("compiler should not run for an existing PRD")

    monkeypatch.setattr(prd_generation, "build_graph", lambda: UnexpectedGraph())

    result = prd_generation.generate_project_prd(PROJECT_ID)

    assert result == {"project_id": PROJECT_ID, "prd": "ready"}


def test_generate_prd_compiles_from_approved_durable_state(monkeypatch):
    _patch_project_boundary(monkeypatch)

    approved = _approved_state()
    compiled = _approved_state(
        prd_contract=object(),
        checkpoint_cursor="phase_complete",
        interview_status="AWAITING_PHASE_CHOICE",
    )
    states = iter([approved, compiled])
    monkeypatch.setattr(
        prd_generation,
        "load_checkpoint",
        lambda session_id: next(states),
    )

    saved = []
    monkeypatch.setattr(
        prd_generation,
        "save_checkpoint",
        lambda state: saved.append(dict(state)),
    )

    invoked = []

    class FakeGraph:
        def invoke(self, state, config):
            invoked.append((dict(state), config))

    monkeypatch.setattr(prd_generation, "build_graph", lambda: FakeGraph())

    result = prd_generation.generate_project_prd(PROJECT_ID)

    assert result == {"project_id": PROJECT_ID, "prd": "ready"}
    assert saved[0]["checkpoint_cursor"] == "compile_prd"
    assert saved[0]["interview_status"] == "COMPILING_PRD"
    assert invoked[0][0]["checkpoint_cursor"] == "compile_prd"
    assert invoked[0][1]["metadata"]["openai_usage_session"] == SESSION_ID


def test_generate_prd_surfaces_compilation_failure_as_retryable(monkeypatch):
    _patch_project_boundary(monkeypatch)

    approved = _approved_state(checkpoint_cursor="compile_prd")
    failed = _approved_state(
        checkpoint_cursor="compile_prd",
        interview_status="COMPILING_PRD",
        compilation_errors=["Verifier rejected unsupported claim."],
    )
    states = iter([approved, failed])
    monkeypatch.setattr(
        prd_generation,
        "load_checkpoint",
        lambda session_id: next(states),
    )

    class FakeGraph:
        def invoke(self, state, config):
            return None

    monkeypatch.setattr(prd_generation, "build_graph", lambda: FakeGraph())

    with pytest.raises(prd_generation.PrdGenerationProcessingError) as exc:
        prd_generation.generate_project_prd(PROJECT_ID)

    assert exc.value.project_id == PROJECT_ID
    assert exc.value.retryable is True
    assert exc.value.reason == "Verifier rejected unsupported claim."
