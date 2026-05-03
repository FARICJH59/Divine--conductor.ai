"""ValidatorAgent — post-consistency semantic integrity checker.

Performs cross-modal heuristic validation on the final enriched shot list.
It acts as the "VLM eye" of the pipeline: scanning each shot for the three
critical failure modes defined by the stress-test framework.

Failure modes checked
---------------------
``TEMPORAL_CONFLICT``
    A shot's prompt still contains motion-blur tokens despite a fast-shutter
    constraint being configured.

``HALLUCINATION``
    A shot's prompt contains genre-inappropriate content (e.g. orange fire or
    VFX fire textures in a biblical production).

``CHARACTER_ANCHOR_DRIFT``
    A shot references a character via ``consistency_anchors`` but none of the
    character's anchor description keywords survive in the final enriched prompt,
    suggesting the character has "merged" with the environment.

All detected failures are:
  1. Appended to ``state.consistency_report`` so they appear in the pipeline output.
  2. Recorded in the optional ``FailureLog`` (if one is provided) so that the
     orchestrator can persist them and suggest ``motion_bucket`` adjustments for
     the next render.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from typing import Any

from divine_conductor.agents.base import BaseAgent
from divine_conductor.models.production import ProductionState, Shot
from divine_conductor.pipeline.failure_log import FailureLog, FailureRecord, FailureType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword libraries used by the heuristic checks
# ---------------------------------------------------------------------------

# Tokens indicating motion blur (should be absent when fast-shutter is set)
_MOTION_BLUR_TOKENS: frozenset[str] = frozenset({
    "motion blur",
    "blur",
    "smear",
    "streaking",
    "long exposure",
})

# Genre-inappropriate content tokens
_HALLUCINATION_TOKENS: dict[str, list[str]] = {
    "biblical": [
        "orange fire",
        "fuel fire",
        "VFX fire",
        "modern explosion",
        "CGI fire",
        "neon",
    ],
}

# Fast-shutter threshold — consistent with ConflictResolverAgent
_FAST_SHUTTER_THRESHOLD = Fraction(1, 500)

# Default motion-bucket deltas per failure type
_MOTION_BUCKET_DELTAS: dict[FailureType, float] = {
    FailureType.TEMPORAL_CONFLICT: -0.1,
    FailureType.HALLUCINATION: 0.0,
    FailureType.CHARACTER_ANCHOR_DRIFT: 0.05,
}


def _is_fast_shutter(shutter: str) -> bool:
    try:
        return Fraction(shutter.strip()) < _FAST_SHUTTER_THRESHOLD
    except (ValueError, ZeroDivisionError):
        return False


class ValidatorAgent(BaseAgent):
    """Semantic integrity validator that runs after the Veo3ConsistencyEngine.

    Args:
        failure_log: Optional ``FailureLog`` to record detected failures into.
            When ``None``, failures are still appended to
            ``state.consistency_report`` but not persisted.
    """

    name = "validator_agent"

    def __init__(self, failure_log: FailureLog | None = None) -> None:
        self._failure_log = failure_log

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, state: ProductionState) -> ProductionState:
        """Validate all shots in *state* and record any failures found."""
        new_issues: list[dict[str, Any]] = []

        for shot in state.shots:
            new_issues.extend(self._check_temporal_conflict(shot, state))
            new_issues.extend(self._check_hallucination(shot, state))
            new_issues.extend(self._check_character_anchor_drift(shot))

        if new_issues:
            logger.warning(
                "[%s] %d semantic issue(s) detected across %d shot(s).",
                self.name,
                len(new_issues),
                len(state.shots),
            )
        else:
            logger.info("[%s] All shots passed semantic validation. ✅", self.name)

        state.consistency_report.extend(new_issues)
        state.metadata["validator_issues_found"] = len(new_issues)
        return state

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_temporal_conflict(
        self, shot: Shot, state: ProductionState
    ) -> list[dict[str, Any]]:
        """Flag shots that contain motion-blur tokens despite a fast-shutter setting."""
        sc = state.config.style_conflict
        if sc is None or not sc.shutter or not _is_fast_shutter(sc.shutter):
            return []

        prompt_lower = shot.prompt.lower()
        leaked = [tok for tok in _MOTION_BLUR_TOKENS if tok in prompt_lower]
        if not leaked:
            return []

        detail = (
            f"Shot {shot.id[:8]}: motion-blur token(s) {leaked!r} found in prompt "
            f"despite fast shutter '{sc.shutter}'. "
            "Freeze-frame constraint violated — check ConflictResolverAgent output."
        )
        logger.warning("[%s] TEMPORAL_CONFLICT: %s", self.name, detail)
        self._log_failure(
            shot=shot,
            failure_type=FailureType.TEMPORAL_CONFLICT,
            detected_value=f"motion blur tokens: {leaked}",
            expected_value="no motion blur tokens (freeze-frame sharp)",
        )
        return [
            {
                "type": FailureType.TEMPORAL_CONFLICT.value,
                "shot_id": shot.id,
                "scene_id": shot.scene_id,
                "detail": detail,
            }
        ]

    def _check_hallucination(
        self, shot: Shot, state: ProductionState
    ) -> list[dict[str, Any]]:
        """Flag shots containing genre-inappropriate content tokens."""
        genre = state.config.genre.lower()
        bad_tokens = _HALLUCINATION_TOKENS.get(genre, [])
        if not bad_tokens:
            return []

        prompt_lower = shot.prompt.lower()
        found = [tok for tok in bad_tokens if tok.lower() in prompt_lower]
        if not found:
            return []

        detail = (
            f"Shot {shot.id[:8]}: genre-inappropriate token(s) {found!r} detected "
            f"in '{genre}' production. AI may have hallucinated VFX-style content."
        )
        logger.warning("[%s] HALLUCINATION: %s", self.name, detail)
        self._log_failure(
            shot=shot,
            failure_type=FailureType.HALLUCINATION,
            detected_value=f"inappropriate tokens: {found}",
            expected_value=f"no genre-inappropriate tokens for '{genre}'",
        )
        return [
            {
                "type": FailureType.HALLUCINATION.value,
                "shot_id": shot.id,
                "scene_id": shot.scene_id,
                "detail": detail,
            }
        ]

    def _check_character_anchor_drift(
        self, shot: Shot
    ) -> list[dict[str, Any]]:
        """Flag shots where a character anchor is present but its description is absent.

        The Veo3ConsistencyEngine injects anchor text into the prompt whenever a
        character ID appears.  If none of the anchor's key descriptive words survive
        in the final prompt, the character has likely "merged" with the environment.
        """
        issues: list[dict[str, Any]] = []
        prompt_lower = shot.prompt.lower()

        for key, anchor_text in shot.consistency_anchors.items():
            if not key.startswith("character:"):
                continue

            # Extract significant words (length > 4) from the anchor description
            anchor_words = {
                w.lower().strip(".,;:")
                for w in str(anchor_text).split()
                if len(w) > 4
            }
            if not anchor_words:
                continue

            if not any(word in prompt_lower for word in anchor_words):
                detail = (
                    f"Shot {shot.id[:8]}: character anchor '{key}' is registered "
                    "but none of its descriptive keywords appear in the final prompt. "
                    "Character may have drifted into the environment."
                )
                logger.warning(
                    "[%s] CHARACTER_ANCHOR_DRIFT: %s", self.name, detail
                )
                self._log_failure(
                    shot=shot,
                    failure_type=FailureType.CHARACTER_ANCHOR_DRIFT,
                    detected_value="anchor keywords absent from prompt",
                    expected_value=f"at least one keyword from: {anchor_words}",
                )
                issues.append(
                    {
                        "type": FailureType.CHARACTER_ANCHOR_DRIFT.value,
                        "shot_id": shot.id,
                        "scene_id": shot.scene_id,
                        "character": key,
                        "detail": detail,
                    }
                )

        return issues

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _log_failure(
        self,
        shot: Shot,
        failure_type: FailureType,
        detected_value: str,
        expected_value: str,
    ) -> None:
        """Record a failure to the ``FailureLog`` if one is configured."""
        if self._failure_log is None:
            return
        self._failure_log.record(
            FailureRecord(
                shot_id=shot.id,
                scene_id=shot.scene_id,
                failure_type=failure_type,
                detected_value=detected_value,
                expected_value=expected_value,
                motion_bucket_delta=_MOTION_BUCKET_DELTAS[failure_type],
            )
        )
