"""Derived implications that sit between confirmed facts and requirements.

Implications are not product facts. They record why the current confirmed model
makes a product-specific requirement relevant, while preserving the fact IDs
that justify the inference.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from agents.discovery_coverage import fact_id
from agents.requirement_activation import ACTIVATION_RULES
from agents.state import AgentState, DiscoveryScope, KnowledgeItem


class ProductImplication(BaseModel):
    id: str
    scope: DiscoveryScope
    rule_id: str
    summary: str
    fact_ids: List[str] = Field(default_factory=list)
    requirement_ids: List[str] = Field(default_factory=list)


def infer_product_implications(
    knowledge: list[KnowledgeItem],
    scope: DiscoveryScope,
) -> list[ProductImplication]:
    implications: list[ProductImplication] = []
    for rule in ACTIVATION_RULES:
        matched = rule.matching_facts(knowledge, scope)
        if not matched:
            continue
        identities = [fact_id(item) for item in matched]
        for template in rule.activates:
            implications.append(ProductImplication(
                id=f"{scope.value}|{rule.id}|{template.id}",
                scope=scope,
                rule_id=rule.id,
                summary=(
                    f"{rule.description} This makes '{template.label}' relevant "
                    "to the current product model."
                ),
                fact_ids=identities,
                requirement_ids=[template.id],
            ))
    return implications


def product_implication_node(state: AgentState) -> dict:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    implications = infer_product_implications(
        state.get("discovered_knowledge", []),
        scope,
    )
    return {
        "model_implications": [
            implication.model_dump(mode="json")
            for implication in implications
        ]
    }
