"""Clean adaptive product-discovery engine.

This package intentionally does not import the legacy discovery planner,
knowledge tracker, schema gaps, or guardrails.  It only reuses the shared
OpenAI client/usage infrastructure from agents.llm.
"""

from .engine import PMDiscoveryEngine
from .models import DiscoveryState

__all__ = ["PMDiscoveryEngine", "DiscoveryState"]
