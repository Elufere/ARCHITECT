"""First-class product concepts discovered from founder language.

These concepts complement the legacy schema-backed KnowledgeItem ledger. They
represent emergent domain structure (entities, relationships, attributes) that
may not have a natural discovery-schema field but is essential for coherent
product reasoning and conversation planning.
"""
from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from agents.state import DiscoveryScope


class ProductConceptKind(str, Enum):
    ENTITY = "ENTITY"
    RELATIONSHIP = "RELATIONSHIP"
    ATTRIBUTE = "ATTRIBUTE"


class ProductConcept(BaseModel):
    kind: ProductConceptKind
    scope: DiscoveryScope
    subject: str
    relation: Optional[str] = None
    object: Optional[str] = None
    value: str
    evidence: str
    confidence: float = Field(ge=0, le=1)
    source_turn: int = 0

    @model_validator(mode="after")
    def semantic_shape(self):
        if not self.subject.strip():
            raise ValueError("Product concept requires a subject")
        if self.kind in {ProductConceptKind.RELATIONSHIP, ProductConceptKind.ATTRIBUTE}:
            if not self.relation or not self.relation.strip():
                raise ValueError(f"{self.kind.value} concept requires relation")
            if not self.object or not self.object.strip():
                raise ValueError(f"{self.kind.value} concept requires object")
        return self


def concept_label(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return value[:80]


def concept_id(concept: ProductConcept) -> str:
    payload = {
        "kind": concept.kind.value,
        "scope": concept.scope.value,
        "subject": concept_label(concept.subject),
        "relation": concept_label(concept.relation or ""),
        "object": concept_label(concept.object or ""),
        "value": concept.value.strip().lower(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def merge_product_concepts(
    existing: List[dict] | List[ProductConcept],
    new: List[ProductConcept],
) -> List[dict]:
    result: List[ProductConcept] = [
        item if isinstance(item, ProductConcept) else ProductConcept.model_validate(item)
        for item in existing
    ]
    known = {concept_id(item) for item in result}
    for item in new:
        identity = concept_id(item)
        if identity in known:
            continue
        known.add(identity)
        result.append(item)
    return [item.model_dump(mode="json") for item in result]
