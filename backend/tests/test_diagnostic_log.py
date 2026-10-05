import logging
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.diagnostic_log import (
    diagnostic_log_path,
    diagnostic_session,
    log_messages_since,
    log_state_snapshot,
    session_log_exists,
    runtime_code_identity,
)


def test_diagnostic_session_appends_terminal_output(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_LOG_DIR", str(tmp_path / "logs"))
    session_id = str(uuid4())

    with diagnostic_session(
        session_id,
        project_id="escrow-app-a1b2c3d4",
        operation="discovery_turn:answer",
    ):
        print("===== DISCOVERY TEST MARKER =====")
        print("planner: selected payment failure recovery")

    path = diagnostic_log_path(session_id)
    content = path.read_text(encoding="utf-8")

    assert session_log_exists(session_id)
    assert "ARCHITECT SESSION LOG" in content
    assert "project: escrow-app-a1b2c3d4" in content
    assert "operation: discovery_turn:answer" in content
    assert "===== DISCOVERY TEST MARKER =====" in content
    assert "planner: selected payment failure recovery" in content
    assert "status: completed" in content


def test_diagnostic_session_preserves_failure_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_LOG_DIR", str(tmp_path / "logs"))
    session_id = str(uuid4())

    with pytest.raises(RuntimeError, match="diagnostic boom"):
        with diagnostic_session(
            session_id,
            project_id="escrow-app-a1b2c3d4",
            operation="discovery_retry",
        ):
            raise RuntimeError("diagnostic boom")

    content = diagnostic_log_path(session_id).read_text(encoding="utf-8")

    assert "ARCHITECT CAPTURED TRACEBACK" in content
    assert "RuntimeError: diagnostic boom" in content
    assert "status: failed (RuntimeError: diagnostic boom)" in content


def test_download_debug_log_route_returns_session_file(tmp_path, monkeypatch):
    import api.routes as routes

    session_id = str(uuid4())
    path = tmp_path / "project-debug.log"
    path.write_text("full session diagnostics", encoding="utf-8")

    monkeypatch.setattr(
        routes,
        "get_project",
        lambda project_id: SimpleNamespace(
            id=project_id,
            discovery_session_id=session_id,
        ),
    )
    monkeypatch.setattr(routes, "session_log_exists", lambda current: True)
    monkeypatch.setattr(routes, "diagnostic_log_path", lambda current: path)

    response = routes.download_project_debug_log("escrow-app-a1b2c3d4")

    assert Path(response.path) == path
    assert "escrow-app-a1b2c3d4-architect-debug.log" in response.headers[
        "content-disposition"
    ]


def test_download_debug_log_route_returns_404_before_first_log(monkeypatch):
    import api.routes as routes
    from fastapi import HTTPException

    session_id = str(uuid4())
    monkeypatch.setattr(
        routes,
        "get_project",
        lambda project_id: SimpleNamespace(
            id=project_id,
            discovery_session_id=session_id,
        ),
    )
    monkeypatch.setattr(routes, "session_log_exists", lambda current: False)

    with pytest.raises(HTTPException) as exc:
        routes.download_project_debug_log("escrow-app-a1b2c3d4")

    assert exc.value.status_code == 404



def test_diagnostic_log_contains_state_conversation_and_python_logging(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ARCHITECT_LOG_DIR", str(tmp_path / "logs"))
    session_id = str(uuid4())

    with diagnostic_session(
        session_id,
        project_id="escrow-app-a1b2c3d4",
        operation="discovery_turn:answer",
    ):
        logging.getLogger("agents.guardrails").warning(
            "Question retry limit reached"
        )
        log_state_snapshot(
            {
                "turn_count": 8,
                "discovery_scope": "USER_APP",
                "checkpoint_cursor": "waiting",
                "interview_status": "WAITING_FOR_USER",
                "conversation_intent": "product_information",
                "current_topic": "CORE_WORKFLOW",
                "current_gap": "workflow_steps",
                "current_objective": "Decide payment recovery.",
                "active_discovery_thread": "payments",
                "selected_inquiry": {
                    "id": "inquiry-payment",
                    "decision_key": "payment_failure_recovery",
                },
                "active_requirements": {
                    "USER_APP|payment_recovery": {"status": "ACTIVE"}
                },
                "open_inquiries": [],
                "discovery_boundaries": [],
                "founder_requested_completion": False,
                "completion_arbitration_complete": False,
                "prd_confirmation_pending": False,
                "ready_to_compile": False,
                "prd_contract": None,
                "extraction_status": "SUCCESS",
            },
            label="after discovery turn",
        )
        log_messages_since(
            {
                "messages": [
                    HumanMessage(content="Leave this to engineering."),
                    AIMessage(content="Understood. I will move on."),
                ]
            },
            start_index=0,
        )

    content = diagnostic_log_path(session_id).read_text(encoding="utf-8")

    assert "Question retry limit reached" in content
    assert "ARCHITECT STATE AFTER DISCOVERY TURN" in content
    assert "decision_key: payment_failure_recovery" in content
    assert "active_requirements: 1" in content
    assert "FOUNDER:" in content
    assert "Leave this to engineering." in content
    assert "ARCHITECT:" in content
    assert "Understood. I will move on." in content



def test_diagnostic_session_records_runtime_code_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARCHITECT_GIT_BRANCH", "feature/test-branch")
    monkeypatch.setenv("ARCHITECT_GIT_SHA", "abc123runtime")
    monkeypatch.setenv("ARCHITECT_GIT_DIRTY", "no")
    runtime_code_identity.cache_clear()
    session_id = str(uuid4())

    with diagnostic_session(
        session_id,
        project_id="todo-test-a1b2c3d4",
        operation="project_initialization",
    ):
        print("revision marker")

    content = diagnostic_log_path(session_id).read_text(encoding="utf-8")
    assert "code_branch: feature/test-branch" in content
    assert "code_revision: abc123runtime" in content
    assert "code_dirty: no" in content
    runtime_code_identity.cache_clear()
