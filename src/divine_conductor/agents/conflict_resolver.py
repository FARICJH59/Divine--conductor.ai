"""ConflictResolverAgent — resolves style conflicts and injects negative prompts.

Detects contradictions between ``StyleConflictMetadata`` (camera settings,
kinetic levels) and the production genre, then injects appropriate negative
prompt tokens into every shot to suppress unwanted AI defaults.

Typical conflicts resolved:

* **Fast shutter vs. slow-creation genre** — a ``1/2000`` shutter implies
  freeze-frame debris while the *biblical* genre defaults to soft, fluid
  motion.  The resolver kills motion-blur to enforce sharp particulate detail.
* **Tectonic kinetic level** — chaotic high-pressure events must not be
  rendered with smooth, liquid-like surfaces.
* **Genre hallucination prevention** — biblical scenes must never show
  orange fuel fire or modern VFX fire textures.

The agent runs *after* ``CinematographerAgent`` (shots already exist) and
*before* the ``Veo3ConsistencyEngine`` so that negative prompts are baked
in before final enrichment.
"""

from __future__ import annotations

import logging
from fractions import Fraction

from divine_conductor.agents.base import BaseAgent
from divine_conductor.models.production import ProductionState, Shot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Negative-prompt libraries
# ---------------------------------------------------------------------------

# Negatives triggered by kinetic level
_KINETIC_NEGATIVES: dict[str, list[str]] = {
    "tectonic_violence": [
        "motion blur",
        "liquid smear",
        "smooth surface",
        "soft edges",
        "fluid motion",
        "dreamy",
        "slow movement",
        "bokeh",
    ],
}

# Negatives triggered by genre
_GENRE_NEGATIVES: dict[str, list[str]] = {
    "biblical": [
        "orange fire",
        "fuel fire",
        "VFX fire",
        "modern explosion",
        "CGI fire",
        "neon light",
        "digital artifacts",
        "lens distortion",
    ],
}

# Shutter speed above this threshold (as a fraction of 1 second) is
# considered "fast" — i.e. the user wants freeze-frame sharpness.
_FAST_SHUTTER_THRESHOLD = Fraction(1, 500)  # faster than 1/500

# Genres that imply a naturally slow/gradual pace (used for conflict detection)
_SLOW_CREATION_GENRES: frozenset[str] = frozenset({"biblical"})


def _parse_shutter(shutter: str) -> Fraction | None:
    """Parse a shutter speed string like ``"1/2000"`` into a ``Fraction``.

    Returns ``None`` if parsing fails.
    """
    try:
        return Fraction(shutter.strip())
    except (ValueError, ZeroDivisionError):
        return None


def _is_fast_shutter(shutter: str) -> bool:
    """Return ``True`` when *shutter* is faster than the fast-shutter threshold."""
    value = _parse_shutter(shutter)
    if value is None:
        return False
    return value < _FAST_SHUTTER_THRESHOLD


class ConflictResolverAgent(BaseAgent):
    """Resolves style conflicts by injecting negative prompts into shots.

    Reads ``state.config.style_conflict`` and ``state.config.genre`` and
    builds a combined set of negative-prompt tokens.  These are appended
    (comma-separated) to every shot's ``negative_prompt`` field.

    If no style conflict metadata is configured the agent is a no-op.

    The agent also logs a warning when a fast shutter speed conflicts with a
    slow-creation genre so that human reviewers are aware of the intentional
    override.
    """

    name = "conflict_resolver_agent"

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, state: ProductionState) -> ProductionState:
        """Inject conflict-derived negative prompts into all shots in *state*."""
        negatives = self._build_negatives(state)
        if not negatives:
            logger.debug("[%s] No style conflicts detected; skipping.", self.name)
            return state

        negative_str = ", ".join(negatives)
        logger.info(
            "[%s] Injecting %d negative constraint(s): %s",
            self.name,
            len(negatives),
            negative_str,
        )

        state.shots = [
            self._apply_negatives(shot, negative_str) for shot in state.shots
        ]
        state.metadata["conflict_resolver_negatives"] = list(negatives)
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_negatives(self, state: ProductionState) -> list[str]:
        """Collect all negative-prompt tokens applicable to this production."""
        negatives: list[str] = []
        sc = state.config.style_conflict
        genre = state.config.genre.lower()

        # Genre-level negatives (always applied regardless of style_conflict)
        negatives.extend(_GENRE_NEGATIVES.get(genre, []))

        if sc is None:
            return negatives

        # Kinetic-level negatives
        kinetic = sc.kinetic_level.lower()
        negatives.extend(_KINETIC_NEGATIVES.get(kinetic, []))

        # Fast-shutter conflict with slow-creation genre
        if sc.shutter and genre in _SLOW_CREATION_GENRES:
            if _is_fast_shutter(sc.shutter):
                logger.warning(
                    "[%s] Temporal conflict detected: shutter '%s' contradicts "
                    "slow-creation genre '%s'. Freeze-frame negatives applied.",
                    self.name,
                    sc.shutter,
                    genre,
                )
                # Additional freeze-frame enforcement
                for neg in ["motion blur", "soft edges"]:
                    if neg not in negatives:
                        negatives.append(neg)

        return negatives

    @staticmethod
    def _apply_negatives(shot: Shot, negative_str: str) -> Shot:
        """Return a new ``Shot`` with *negative_str* appended to ``negative_prompt``."""
        existing = shot.negative_prompt.strip()
        merged = f"{existing}, {negative_str}" if existing else negative_str
        return Shot(
            scene_id=shot.scene_id,
            index=shot.index,
            prompt=shot.prompt,
            negative_prompt=merged,
            duration_seconds=shot.duration_seconds,
            camera_angle=shot.camera_angle,
            consistency_anchors=dict(shot.consistency_anchors),
            id=shot.id,
        )
