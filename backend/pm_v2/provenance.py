"""Deterministic provenance checks only.

This module deliberately does not decide what words mean.  It verifies that
model-proposed evidence came from the submitted user turn and records offsets.
Semantic support is audited by the grounding LLM.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSpan:
    text: str
    start: int
    end: int


def locate_exact_evidence(source: str, evidence: str) -> EvidenceSpan | None:
    if not source or not evidence:
        return None
    start = source.find(evidence)
    if start < 0:
        return None
    return EvidenceSpan(text=evidence, start=start, end=start + len(evidence))


def provenance_valid(source: str, evidence: str) -> bool:
    return locate_exact_evidence(source, evidence) is not None
