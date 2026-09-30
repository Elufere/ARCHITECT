from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import HumanMessage

from agents import external_systems as systems
from agents import knowledge_tracker as tracker
from agents.external_systems import (
    ExternalSystemGrounding,
    ExternalSystemMention,
    external_system_context,
    merge_external_systems,
)
from agents.pm_agent import build_prd_external_systems
from agents.product_model import build_product_model
from agents.state import DiscoveryScope
from agents.understanding_projection import build_understanding_projection
from agents.interview_checkpoint import load_checkpoint, save_checkpoint
from services.discovery_session import create_initial_discovery_state


SCOPE = DiscoveryScope.USER_APP


def mention(
    value="Payments go through Flutterwave.",
    *,
    evidence=None,
    relation="processes",
    object="payments",
    turn=3,
):
    return ExternalSystemMention(
        scope=SCOPE,
        name="Flutterwave",
        value=value,
        evidence=evidence or value,
        relation=relation,
        object=object,
        confidence=1,
        source_turn=turn,
    )


def test_external_system_mentions_merge_under_one_identity():
    merged = merge_external_systems([], [mention()])
    merged = merge_external_systems(
        merged,
        [
            mention(
                "Flutterwave verifies the payment before the deal continues.",
                relation="verifies",
                object="payment",
                turn=4,
            )
        ],
    )

    assert len(merged) == 1
    system = merged[0]
    assert system["name"] == "Flutterwave"
    assert system["id"].startswith("external:")
    assert [item["relation"] for item in system["statements"]] == [
        "processes",
        "verifies",
    ]


def test_external_system_merge_is_idempotent():
    first = mention()
    merged = merge_external_systems([], [first])
    merged = merge_external_systems(merged, [first])

    assert len(merged) == 1
    assert len(merged[0]["statements"]) == 1


def test_external_system_grounding_keeps_only_supported_candidates(monkeypatch):
    candidates = [
        mention(),
        ExternalSystemMention(
            scope=SCOPE,
            name="Acme",
            value="Acme was mentioned.",
            evidence="Acme was mentioned.",
            confidence=1,
            source_turn=3,
        ),
    ]

    class FakeGrounder:
        def invoke(self, messages):
            return ExternalSystemGrounding(
                supported_ids=[0],
                rejection_reasons={"1": "A name alone does not establish an integration."},
            )

    monkeypatch.setattr(systems, "external_system_grounder", lambda: FakeGrounder())

    grounded = systems.ground_external_system_mentions(
        candidates,
        "Payments go through Flutterwave. Acme was mentioned.",
        {
            "messages": [
                HumanMessage(
                    content="Payments go through Flutterwave. Acme was mentioned."
                )
            ],
            "discovery_scope": SCOPE,
            "external_systems": [],
        },
    )

    assert [item.name for item in grounded] == ["Flutterwave"]


def test_claim_capture_keeps_flutterwave_out_of_actor_and_product_entity_ledgers(monkeypatch):
    text = "We will use Flutterwave for escrow payments."

    class FakeCapture:
        def invoke(self, messages):
            return {
                "parsed": {
                    "items": [
                        {
                            "kind": "external_system",
                            "value": text,
                            "evidence": text,
                            "subject": "Flutterwave",
                            "relation": "handles",
                            "object": "escrow payments",
                            "confidence": 1,
                            "knowledge_state": "CONFIRMED",
                        }
                    ]
                }
            }

    monkeypatch.setattr(
        tracker,
        "extraction_models",
        lambda: {"CLAIMS": FakeCapture()},
    )
    monkeypatch.setattr(
        tracker,
        "ground_external_system_mentions",
        lambda mentions, response, state: mentions,
    )

    batch = tracker.extract_passes(
        text,
        {
            "messages": [HumanMessage(content=text)],
            "discovery_scope": SCOPE,
            "discovered_knowledge": [],
            "external_systems": [],
            "current_topic": None,
            "current_gap": None,
            "turn_count": 1,
        },
        SCOPE,
    )

    assert list(batch) == []
    assert batch.concepts == []
    assert len(batch.external_systems) == 1
    assert batch.external_systems[0].name == "Flutterwave"


def test_external_system_context_exposes_only_current_scope():
    user_systems = merge_external_systems([], [mention()])
    admin_system = ExternalSystemMention(
        scope=DiscoveryScope.ADMIN_DASHBOARD,
        name="Internal BI",
        value="The admin dashboard sends data to Internal BI.",
        evidence="The admin dashboard sends data to Internal BI.",
        confidence=1,
        source_turn=4,
    )
    all_systems = merge_external_systems(user_systems, [admin_system])

    context = external_system_context(
        {"external_systems": all_systems},
        SCOPE,
    )

    assert [item["name"] for item in context] == ["Flutterwave"]


def test_understanding_projection_has_dedicated_external_system_section():
    external = merge_external_systems([], [mention()])
    projection = build_understanding_projection({
        "discovery_scope": SCOPE,
        "discovered_knowledge": [],
        "product_concepts": [],
        "external_systems": external,
        "active_requirements": {},
        "requirement_coverage": {},
        "requirement_dependency_state": {},
        "open_inquiries": [],
        "validation_issues": [],
        "discovery_boundaries": [],
    })

    sections = {section.id: section for section in projection.sections}

    assert "external_systems" in sections
    assert sections["external_systems"].title == "External systems & integrations"
    assert sections["external_systems"].items[0].label == "Flutterwave"
    assert "Payments go through Flutterwave." in (
        sections["external_systems"].items[0].detail or ""
    )
    assert "product_structure" not in sections


def test_product_model_exposes_external_systems_to_planning():
    external = merge_external_systems([], [mention()])

    model = build_product_model([], SCOPE, [], external)

    assert model["EXTERNAL_SYSTEMS"] == [
        "EXTERNAL SYSTEM Flutterwave: Payments go through Flutterwave."
    ]


def test_external_system_survives_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    state = create_initial_discovery_state(
        "An escrow product.",
        session_id=str(uuid4()),
    )
    state["external_systems"] = merge_external_systems([], [mention()])
    save_checkpoint(state)

    restored = load_checkpoint(state["session_id"])

    assert restored["external_systems"][0]["name"] == "Flutterwave"
    assert restored["external_systems"][0]["statements"][0]["object"] == "payments"


def test_legacy_checkpoint_without_external_systems_gets_empty_ledger(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    state = create_initial_discovery_state(
        "An escrow product.",
        session_id=str(uuid4()),
    )
    state.pop("external_systems")
    save_checkpoint(state)

    restored = load_checkpoint(state["session_id"])

    assert restored["external_systems"] == []


def test_prd_projection_carries_grounded_external_systems_without_llm_rewrite():
    external = merge_external_systems([], [mention()])
    contract_systems = build_prd_external_systems(
        {"external_systems": external},
        SCOPE,
    )

    assert len(contract_systems) == 1
    assert contract_systems[0].name == "Flutterwave"
    assert contract_systems[0].statements[0].evidence == (
        "Payments go through Flutterwave."
    )
    assert contract_systems[0].statements[0].relation == "processes"
