"""
Pydantic schemas for the Product Manager Agent output.
Ensures the LLM outputs a deterministic, machine-readable contract.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ScopeBoundary(BaseModel):
    in_scope: list[str] = Field(description="Explicit features and functionalities included in the MVP.")
    out_of_scope: list[str] = Field(description="Explicit features deferred to future versions.")


class UserPersona(BaseModel):
    name: str = Field(description="Name of the user persona (e.g., 'End Customer', 'Admin').")
    description: str = Field(description="Brief description of who they are and their goal.")
    key_behaviors: list[str] = Field(description="List of core interactions they will perform.")


class FunctionalRequirement(BaseModel):
    id: str = Field(description="Unique identifier, e.g., 'FR-01'.")
    description: str = Field(description="The system SHALL [do something]...")
    category: str = Field(description="e.g., 'Authentication', 'Data Management', 'API'.")
    validation: Optional[str] = Field(default=None, description="How do we know this requirement is met?")


class PRDContract(BaseModel):
    """
    The final, validated output of the PM Agent.
    This object is saved to disk and passed to the Architect Agent.
    """
    product_name: str = Field(description="A concise, internal name for the product.")
    elevator_pitch: str = Field(description="A 2-3 sentence summary of what the product does and for whom.")
    
    scope: ScopeBoundary = Field(description="Strict boundaries of the MVP.")
    personas: list[UserPersona] = Field(description="Primary users of the system.")
    
    functional_requirements: list[FunctionalRequirement] = Field(
        description="Detailed, testable system behaviors."
    )
    
    non_functional_constraints: list[str] = Field(
        default_factory=list,
        description="Performance, security, or compliance notes derived from KB rules."
    )