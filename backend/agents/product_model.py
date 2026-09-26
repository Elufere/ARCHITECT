"""A shared, derived view of product knowledge across all discovery topics."""

from collections import defaultdict
from typing import Dict, Iterable, List

from agents.state import DiscoveryScope, KnowledgeItem, KnowledgeState
from agents.product_concepts import ProductConcept


def build_product_model(
    knowledge: Iterable[KnowledgeItem],
    scope: DiscoveryScope,
    product_concepts: Iterable[dict | ProductConcept] | None = None,
) -> Dict[str, List[str]]:
    model: dict[str, list[str]] = defaultdict(list)
    for item in knowledge:
        if item.scope != scope or item.knowledge_state != KnowledgeState.CONFIRMED:
            continue
        label = f"{item.topic.value}.{item.key}"
        if item.role:
            label += f"[{item.role}]"
        fact = f"{label}: {item.value}"
        if fact not in model[item.topic.value]:
            model[item.topic.value].append(fact)
    for raw in product_concepts or []:
        concept = raw if isinstance(raw, ProductConcept) else ProductConcept.model_validate(raw)
        if concept.scope != scope:
            continue
        if concept.kind.value == "ENTITY":
            text = f"ENTITY {concept.subject}: {concept.value}"
        else:
            text = (
                f"{concept.kind.value} {concept.subject} "
                f"-[{concept.relation}]-> {concept.object}: {concept.value}"
            )
        if text not in model["PRODUCT_STRUCTURE"]:
            model["PRODUCT_STRUCTURE"].append(text)
    return dict(model)


def format_product_model(model: Dict[str, List[str]]) -> str:
    if not model:
        return "No confirmed product information yet."
    return "\n\n".join(
        f"{topic}:\n" + "\n".join(f"- {fact}" for fact in facts)
        for topic, facts in model.items()
    )
