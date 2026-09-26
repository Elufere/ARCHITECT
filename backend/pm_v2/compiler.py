from __future__ import annotations

import json
from pathlib import Path

from .context import active_fact_payload
from .llm import structured_call
from .models import DiscoveryState, KnowledgeStatus
from .prd_models import ImplementationReadyPRD


COMPILER_PROMPT = """
Compile an implementation-ready PRD from the supplied discovery state.

Authority rules:
- Confirmed active founder facts are the only source of confirmed product claims.
- Every factual statement and compiled requirement must cite supporting source_fact_ids.
- Proposed implications must remain visibly proposed and may not be written as facts.
- Deferred decisions remain deferred.
- Unknown matters become open questions, not invented requirements.
- Do not reconstruct or rely on the raw conversation.
- Do not add standard industry behavior, security, workflows, admin features, or
  integrations unless confirmed facts support them.
- Preserve actors, negation, conditions, boundaries, money movement, lifecycle,
  and scope exactly.

Organize sections according to the product actually discovered. Useful section
names can include actors, product_model, domain_model, workflows, business_rules,
payments, fulfillment, lifecycle, integrations, constraints, failures, mvp, and
success_metrics, but do not create empty/generic sections merely to fill a template.
"""


def compile_prd(state: DiscoveryState) -> ImplementationReadyPRD:
    if not state.complete:
        raise ValueError("Discovery is not complete enough to compile the PRD.")

    facts = active_fact_payload(state)
    valid_fact_ids = {item["id"] for item in facts}
    confirmed_requirements = [
        {
            "key": req.key,
            "label": req.label,
            "description": req.description,
            "supporting_fact_ids": [
                identity
                for identity in req.evidence_fact_ids
                if identity in valid_fact_ids
            ],
        }
        for req in state.requirements.values()
        if req.status == KnowledgeStatus.CONFIRMED
    ]
    deferred = [
        req.key
        for req in state.requirements.values()
        if req.status == KnowledgeStatus.DEFERRED
    ]
    unresolved = [
        {
            "key": req.key,
            "missing_decisions": req.missing_decisions,
        }
        for req in state.requirements.values()
        if req.status in {KnowledgeStatus.UNKNOWN, KnowledgeStatus.PROPOSED}
    ]
    proposed_implications = [
        item.statement
        for item in state.implications
        if item.status == KnowledgeStatus.PROPOSED
    ]

    result = structured_call(
        call_name="prd_compile",
        schema=ImplementationReadyPRD,
        instruction=COMPILER_PROMPT,
        payload={
            "confirmed_facts": facts,
            "confirmed_requirements": confirmed_requirements,
            "deferred_requirement_keys": deferred,
            "unresolved_requirements": unresolved,
            "proposed_implications": proposed_implications,
        },
        max_tokens=5000,
    )
    _validate_sources(result, valid_fact_ids)
    return result


def _validate_sources(prd: ImplementationReadyPRD, valid_fact_ids: set[str]) -> None:
    sourced = [
        *prd.product_summary,
        *[
            item
            for section in prd.sections.values()
            for item in section
        ],
    ]
    for item in sourced:
        if not item.source_fact_ids:
            raise ValueError(f"Unsourced PRD statement: {item.statement}")
        if any(identity not in valid_fact_ids for identity in item.source_fact_ids):
            raise ValueError(f"PRD statement cites unknown fact ID: {item.statement}")

    for requirement in prd.requirements:
        if not requirement.source_fact_ids:
            raise ValueError(f"Unsourced PRD requirement: {requirement.key}")
        if any(identity not in valid_fact_ids for identity in requirement.source_fact_ids):
            raise ValueError(f"PRD requirement cites unknown fact ID: {requirement.key}")


def save_prd(prd: ImplementationReadyPRD, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(prd.model_dump_json(indent=2), encoding="utf-8")
    return output
