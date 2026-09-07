"""A shared, derived view of product knowledge across all discovery topics."""

from collections import defaultdict
from typing import Dict, Iterable, List

from agents.state import DiscoveryScope, KnowledgeItem, KnowledgeState


def build_product_model(
    knowledge: Iterable[KnowledgeItem], scope: DiscoveryScope
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
    return dict(model)


def format_product_model(model: Dict[str, List[str]]) -> str:
    if not model:
        return "No confirmed product information yet."
    return "\n\n".join(
        f"{topic}:\n" + "\n".join(f"- {fact}" for fact in facts)
        for topic, facts in model.items()
    )
