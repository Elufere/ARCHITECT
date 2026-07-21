"""
Pydantic schemas for the Product Manager Agent output.
"""

from pydantic import BaseModel, Field

class ScopeBoundary(BaseModel):
    in_scope: list[str] = Field(description="Explicit features included in the MVP.")
    out_of_scope: list[str] = Field(description="Explicit features deferred to future versions.")


class UserPersona(BaseModel):
    name: str = Field(description="e.g., 'End Customer', 'Admin'.")
    description: str = Field(description="Brief description of who they are and their goal.")
    key_behaviors: list[str] = Field(description="Core interactions they will perform.")


class FunctionalRequirement(BaseModel):
    id: str = Field(description="Unique identifier, e.g., 'FR-01'.")
    description: str = Field(description="The system SHALL [do something]...")
    category: str = Field(description="e.g., 'Authentication', 'Data Management'.")
    validation: str = Field(description="How do we know this is met? If unknown, write 'TBD'.")


class PRDContract(BaseModel):
    """The final, validated output of the PM Agent."""
    product_name: str = Field(description="Concise internal name for the product.")
    elevator_pitch: str = Field(description="2-3 sentence summary of the product.")
    
    scope: ScopeBoundary = Field(description="Strict boundaries of the MVP.")
    personas: list[UserPersona] = Field(description="Primary users of the system.")
    
    functional_requirements: list[FunctionalRequirement] = Field(
        description="Detailed, testable system behaviors."
    )
    
    non_functional_constraints: list[str] = Field(
        description="Performance, security, or compliance notes."
    )
    
    # NEW: Structural fixes for unconfirmed items
    deferred_items: list[str] = Field(
        default_factory=list,
        description="Features raised by user or AI but explicitly marked as 'future/v2/out of scope'."
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Critical questions that remain unanswered. Architect Agent must note these."
    )