from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SourcedStatement(BaseModel):
    statement: str
    source_fact_ids: List[str] = Field(default_factory=list)


class CompiledRequirement(BaseModel):
    key: str
    label: str
    decision: str
    source_fact_ids: List[str] = Field(default_factory=list)


class ImplementationReadyPRD(BaseModel):
    product_name: Optional[str] = None
    product_summary: List[SourcedStatement] = Field(default_factory=list)
    sections: Dict[str, List[SourcedStatement]] = Field(default_factory=dict)
    requirements: List[CompiledRequirement] = Field(default_factory=list)
    deferred_decisions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    proposed_implications: List[str] = Field(default_factory=list)
    proposed_recommendations: List[str] = Field(default_factory=list)
