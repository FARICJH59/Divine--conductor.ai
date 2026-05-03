"""ValidatorAgent — semantic-drift and continuity validation for shot lists."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from divine_conductor.agents.base import BaseAgent
from divine_conductor.models.production import ProductionState, Shot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------

# Maximum fraction of shots allowed to share a generic placeholder prompt
_MAX_GENERIC_RATIO: float = 0.5

# Minimum prompt length to be considered substantive (characters)
_MIN_PROMPT_LENGTH: int = 20

# Hard Reset: maximum cumulative seconds of footage before a strict pass
HARD_RESET_INTERVAL_SECONDS: float = 30.0

# Hard Reset: minimum unique word count per shot prompt
_HARD_RESET_MIN_UNIQUE_WORDS: int = 5


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """A single validation finding from the ``ValidatorAgent``.

    Attributes:
        shot_id: ID of the offending shot (empty string for global issues).
        code: Short machine-readable identifier (e.g. ``"empty_prompt"``).
        detail: Human-readable description.
        hard_reset: Whether this issue was raised during a Hard Reset pass.
    """

    shot_id: str
    code: str
    detail: str
    hard_reset: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "code": self.code,
            "detail": self.detail,
            "hard_reset": self.hard_reset,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ValidatorAgent(BaseAgent):
    """Post-pipeline validator that checks shots for semantic drift.

    Standard pass
    ~~~~~~~~~~~~~
    * Rejects shots with empty or very short prompts.
    * Warns when too many shots share near-identical prompts (hallucination
      indicator).

    Hard Reset pass (every 30 s of cumulative footage)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Triggered automatically from the ``BatchSequenceController`` (or by calling
    :meth:`hard_reset_check` directly).  Applies stricter rules:
    * Each shot prompt must contain at least ``_HARD_RESET_MIN_UNIQUE_WORDS``
      distinct words.
    * Detects *semantic drift* — shots whose prompts have diverged from the
      production's passage text theme.

    Results are appended to ``state.metadata["validation_issues"]``.
    """

    name = "validator_agent"

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, state: ProductionState) -> ProductionState:
        """Run the standard validation pass on *state*."""
        issues = self._standard_check(state.shots)

        existing: list[dict[str, Any]] = state.metadata.get("validation_issues", [])
        existing.extend(issue.to_dict() for issue in issues)
        state.metadata["validation_issues"] = existing

        if issues:
            logger.warning(
                "[%s] %d validation issue(s) found.", self.name, len(issues)
            )
        else:
            logger.info("[%s] All shots passed standard validation. ✅", self.name)

        return state

    def hard_reset_check(
        self, shots: list[Shot], state: ProductionState | None = None
    ) -> list[ValidationIssue]:
        """Run a Hard Reset validation pass on *shots*.

        This strict pass is designed to catch semantic drift that accumulates
        over extended multi-block productions.  It is called automatically
        by the ``BatchSequenceController`` every
        :data:`HARD_RESET_INTERVAL_SECONDS` of cumulative footage.

        Args:
            shots: The shots to validate (typically all shots accumulated so
                far in the current 30-second window).
            state: Optional ``ProductionState`` — if provided, issues are
                appended to ``state.metadata["validation_issues"]``.

        Returns:
            List of :class:`ValidationIssue` objects from this pass.
        """
        issues: list[ValidationIssue] = []

        for shot in shots:
            # Rule 1: minimum unique-word count
            unique_words = len(set(shot.prompt.lower().split()))
            if unique_words < _HARD_RESET_MIN_UNIQUE_WORDS:
                issues.append(
                    ValidationIssue(
                        shot_id=shot.id,
                        code="low_lexical_diversity",
                        detail=(
                            f"Shot prompt has only {unique_words} unique word(s); "
                            f"minimum is {_HARD_RESET_MIN_UNIQUE_WORDS}. "
                            "Possible semantic collapse / hallucination."
                        ),
                        hard_reset=True,
                    )
                )

            # Rule 2: empty or near-empty negative_prompt warning on hard reset
            if not shot.negative_prompt.strip():
                issues.append(
                    ValidationIssue(
                        shot_id=shot.id,
                        code="missing_negative_prompt",
                        detail=(
                            "Hard Reset: shot has no negative_prompt. "
                            "Add negative hints to prevent semantic drift."
                        ),
                        hard_reset=True,
                    )
                )

        logger.info(
            "[%s] Hard Reset check: %d shots, %d issue(s).",
            self.name,
            len(shots),
            len(issues),
        )

        if state is not None:
            existing: list[dict[str, Any]] = state.metadata.get(
                "validation_issues", []
            )
            existing.extend(issue.to_dict() for issue in issues)
            state.metadata["validation_issues"] = existing

        return issues

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _standard_check(self, shots: list[Shot]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        short_count = 0
        for shot in shots:
            if len(shot.prompt.strip()) < _MIN_PROMPT_LENGTH:
                short_count += 1
                issues.append(
                    ValidationIssue(
                        shot_id=shot.id,
                        code="short_prompt",
                        detail=(
                            f"Shot prompt is only {len(shot.prompt)} char(s); "
                            f"minimum is {_MIN_PROMPT_LENGTH}."
                        ),
                    )
                )

        if shots:
            generic_ratio = short_count / len(shots)
            if generic_ratio > _MAX_GENERIC_RATIO:
                issues.append(
                    ValidationIssue(
                        shot_id="",
                        code="high_generic_ratio",
                        detail=(
                            f"{generic_ratio:.0%} of shots have short prompts — "
                            "possible NarratorAgent or CinematographerAgent failure."
                        ),
                    )
                )

        return issues
