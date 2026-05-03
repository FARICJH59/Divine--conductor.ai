"""ValidatorAgent — enforces "Physically Legible Chaos" on shot prompts.

Runs **after** the ``Veo3ConsistencyEngine`` has enriched all shot prompts.
It applies a three-point failure-detection checklist to every shot:

1. **Blur Check** — the prompt must NOT contain words like "smooth", "flowing",
   or "serene" when a ``StyleConflictMetadata`` is active.  These terms
   indicate that the kinetic override leaked and failed to remove fluid motion.

2. **Particle Check** — when a ``StyleConflictMetadata`` is active the prompt
   MUST contain at least one of "debris", "shards", or "fragmentation".
   Their absence means the action physics were never injected (system failure).

3. **Consistency Check** — palette anchor text that was locked by the
   ``Veo3ConsistencyEngine`` must still be detectable in the enriched prompt,
   confirming the colour grade survived the conflict-resolution pass.

Validation results are written to each shot's ``consistency_anchors`` dict
under ``"validation:*"`` keys, and failures are appended to
``state.consistency_report`` with ``category="validation"``.

Summary counters are stored in ``state.metadata`` under ``"validator_*"`` keys.
"""

from __future__ import annotations

import logging
from typing import Any

from divine_conductor.agents.base import BaseAgent
from divine_conductor.models.production import ProductionState, Shot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

# Blur-contamination keywords: their presence in a conflict-active shot
# indicates the kinetic override failed to eliminate fluid/slow-motion rendering.
_BLUR_KEYWORDS: list[str] = ["smooth", "flowing", "serene"]

# Particle keywords: at least one must appear in every conflict-active shot to
# confirm the action-physics layer was successfully injected.
_PARTICLE_KEYWORDS: list[str] = ["debris", "shards", "fragmentation"]

# Result constants
_PASS = "PASS"
_FAIL = "FAIL"


class ValidatorAgent(BaseAgent):
    """Validates shot prompts against the Physically Legible Chaos checklist.

    This agent is automatically inserted into the pipeline by
    ``PipelineOrchestrator`` **after** the ``Veo3ConsistencyEngine`` enrichment
    step, so it operates on the final, fully-enriched prompt strings.

    Args:
        strict_particle_check: When ``True`` the particle check is enforced
            (reports a failure) even when no ``StyleConflictMetadata`` is set.
            Defaults to ``False`` — the check is only enforced when a style
            conflict is active.
    """

    name = "validator_agent"

    def __init__(self, strict_particle_check: bool = False) -> None:
        self._strict_particle = strict_particle_check

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, state: ProductionState) -> ProductionState:
        """Validate all shots in *state* and record results."""
        conflict_active = state.config.style_conflict is not None
        particle_required = conflict_active or self._strict_particle

        new_failures: list[dict[str, Any]] = []

        for shot in state.shots:
            blur_result, blur_violations = self._blur_check(shot, conflict_active)
            particle_result, particle_hits = self._particle_check(
                shot, particle_required
            )
            consistency_result = self._consistency_check(shot, state)

            # Write results into the shot's consistency_anchors dict
            shot.consistency_anchors["validation:blur_check"] = (
                _PASS if blur_result else f"{_FAIL}: {blur_violations}"
            )
            shot.consistency_anchors["validation:particle_check"] = (
                _PASS
                if particle_result
                else (
                    f"{_FAIL}: no particle keywords"
                    if particle_required
                    else f"INFO: no particle keywords (conflict not active)"
                )
            )
            shot.consistency_anchors["validation:consistency_check"] = (
                _PASS if consistency_result else f"{_FAIL}: palette anchor missing"
            )

            # Collect failures for the consistency_report
            if conflict_active and not blur_result:
                new_failures.append(
                    {
                        "shot_id": shot.id,
                        "category": "validation",
                        "check": "blur_check",
                        "detail": (
                            f"Shot {shot.id[:8]}: blur contamination detected — "
                            f"{blur_violations}. Kinetic override did not fully "
                            "suppress fluid motion rendering."
                        ),
                    }
                )
            if particle_required and not particle_result:
                new_failures.append(
                    {
                        "shot_id": shot.id,
                        "category": "validation",
                        "check": "particle_check",
                        "detail": (
                            f"Shot {shot.id[:8]}: particle keywords absent. "
                            "Action-physics layer was not injected — "
                            "system failure (shallow output)."
                        ),
                    }
                )
            if not consistency_result:
                new_failures.append(
                    {
                        "shot_id": shot.id,
                        "category": "validation",
                        "check": "consistency_check",
                        "detail": (
                            f"Shot {shot.id[:8]}: palette anchor missing from "
                            "enriched prompt. Colour grade did not survive "
                            "conflict-resolution pass."
                        ),
                    }
                )

        state.consistency_report = state.consistency_report + new_failures

        shots_checked = len(state.shots)
        state.metadata["validator_shots_checked"] = shots_checked
        state.metadata["validator_failures"] = len(new_failures)

        if new_failures:
            logger.warning(
                "[%s] %d validation failure(s) across %d shot(s).",
                self.name,
                len(new_failures),
                shots_checked,
            )
        else:
            logger.info(
                "[%s] All %d shot(s) passed validation. ✅",
                self.name,
                shots_checked,
            )

        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _blur_check(
        shot: Shot, conflict_active: bool
    ) -> tuple[bool, list[str]]:
        """Return (passed, violations).

        When *conflict_active* is False the check is skipped (always passes)
        because blur keywords are legitimate in non-conflict shots.
        """
        if not conflict_active:
            return True, []
        lower = shot.prompt.lower()
        violations = [kw for kw in _BLUR_KEYWORDS if kw in lower]
        return len(violations) == 0, violations

    @staticmethod
    def _particle_check(
        shot: Shot, particle_required: bool
    ) -> tuple[bool, list[str]]:
        """Return (passed, found_keywords).

        When *particle_required* is False, missing particle keywords are not
        treated as a failure — the check is purely informational.
        """
        lower = shot.prompt.lower()
        hits = [kw for kw in _PARTICLE_KEYWORDS if kw in lower]
        if not particle_required:
            return True, hits  # informational only — never fail
        return len(hits) > 0, hits

    @staticmethod
    def _consistency_check(shot: Shot, state: ProductionState) -> bool:
        """Verify the palette anchor text survived enrichment.

        The palette description is always injected by the consistency engine,
        so its absence in the final prompt indicates a pipeline regression.
        """
        palette_anchor = shot.consistency_anchors.get("palette", "")
        if not palette_anchor:
            # Palette was never applied — skip (no anchor to verify)
            return True
        # A meaningful substring of the palette description should appear
        # in the enriched prompt
        words = palette_anchor.split()
        if not words:
            return True  # whitespace-only anchor — nothing to verify
        first_word = words[0].lower()
        return first_word in shot.prompt.lower()
