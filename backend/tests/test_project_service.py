import pytest
from langchain_core.messages import AIMessage

from agents.interview_checkpoint import load_checkpoint, save_checkpoint
from api.schemas import CreateProjectInput
from services.project_repository import create_project_record, get_project
from services.project_service import create_project_workspace, list_project_summaries
from services.discovery_session import create_initial_discovery_state


def _checkpoint(description: str, *, awaiting_confirmation=False, prd_contract=None):
    state = create_initial_discovery_state(description)
    state["checkpoint_cursor"] = "waiting"
    state["interview_status"] = "WAITING_FOR_USER"
    state["awaiting_confirmation"] = awaiting_confirmation
    state["prd_contract"] = prd_contract
    save_checkpoint(state)
    return state


def test_create_project_workspace_persists_project_and_session_before_initial_discovery(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ARCHITECT_PROJECT_DIR", str(tmp_path / "projects"))

    observed = {}

    def fake_advance(session_id):
        project = get_project(observed["project_id"])
        assert project.discovery_session_id == session_id
        state = load_checkpoint(session_id)
        assert state["raw_idea"] == "A buyer and seller escrow product."
        state["messages"].append(
            AIMessage(
                content="Who are the main users?",
                id="architect-first",
                additional_kwargs={"created_at": "2026-09-30T08:00:00+00:00"},
            )
        )
        state["checkpoint_cursor"] = "waiting"
        state["interview_status"] = "WAITING_FOR_USER"
        save_checkpoint(state)
        return state

    # Capture the generated project id as soon as the repository creates it.
    import services.project_service as service
    original_create_record = service.create_project_record

    def capture_record(**kwargs):
        record = original_create_record(**kwargs)
        observed["project_id"] = record.id
        return record

    monkeypatch.setattr(service, "create_project_record", capture_record)
    monkeypatch.setattr(service, "advance_discovery_to_waiting", fake_advance)

    workspace = create_project_workspace(
        name="Escrow App",
        description="A buyer and seller escrow product.",
    )

    project = get_project(workspace.project.id)
    assert project.name == "Escrow App"
    assert project.description == "A buyer and seller escrow product."
    assert workspace.project.id == project.id
    assert workspace.discovery.activePrompt == "Who are the main users?"
    assert [message.role for message in workspace.discovery.messages] == [
        "founder",
        "architect",
    ]


def test_list_projects_derives_status_from_linked_checkpoint_and_sorts_by_live_update(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ARCHITECT_PROJECT_DIR", str(tmp_path / "projects"))

    first = _checkpoint("First idea")
    first_project = create_project_record(
        name="First",
        description="First idea",
        discovery_session_id=first["session_id"],
    )

    second = _checkpoint("Second idea", awaiting_confirmation=True)
    second_project = create_project_record(
        name="Second",
        description="Second idea",
        discovery_session_id=second["session_id"],
    )

    # Touch the first checkpoint after both Project records exist, proving list
    # order/status come from current workspace activity rather than Project JSON alone.
    first_loaded = load_checkpoint(first_project.discovery_session_id)
    first_loaded["messages"].append(
        AIMessage(content="Latest question", id="latest-question")
    )
    save_checkpoint(first_loaded)

    summaries = list_project_summaries()
    by_id = {item.id: item for item in summaries}

    assert by_id[first_project.id].status == "discovering"
    assert by_id[second_project.id].status == "ready_for_prd"
    assert summaries[0].id == first_project.id


def test_create_project_input_trims_values():
    payload = CreateProjectInput(
        name="  Escrow App  ",
        description="  A secure transaction product.  ",
    )
    assert payload.name == "Escrow App"
    assert payload.description == "A secure transaction product."


def test_create_project_input_rejects_blank_values():
    with pytest.raises(ValueError):
        CreateProjectInput(name="   ", description="idea")
    with pytest.raises(ValueError):
        CreateProjectInput(name="Project", description="   ")
