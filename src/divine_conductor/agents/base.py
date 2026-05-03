"""Abstract base class for all pipeline agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from divine_conductor.models.production import ProductionState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all Divine Conductor agents.

    Sub-classes must implement :meth:`run`, which receives the current
    ``ProductionState``, enriches it in place, and returns it.

    Example::

        class MyAgent(BaseAgent):
            name = "my_agent"

            def run(self, state: ProductionState) -> ProductionState:
                # ... modify state ...
                return state
    """

    #: Human-readable agent name — override in sub-classes.
    name: str = "base_agent"

    def __call__(self, state: ProductionState) -> ProductionState:
        """Invoke the agent, with pre/post logging."""
        logger.info("[%s] Starting.", self.name)
        result = self.run(state)
        logger.info(
            "[%s] Done. scenes=%d shots=%d",
            self.name,
            len(result.scenes),
            len(result.shots),
        )
        return result

    @abstractmethod
    def run(self, state: ProductionState) -> ProductionState:
        """Process *state* and return an enriched copy or the mutated object.

        Args:
            state: The current production state flowing through the pipeline.

        Returns:
            The (possibly mutated) production state.
        """
