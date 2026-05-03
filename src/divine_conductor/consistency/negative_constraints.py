"""Negative constraint management for genre-aware hallucination suppression.

The NegativeConstraintManager builds a negative-prompt string that locks out
anachronistic or tonally-inconsistent visual elements based on the active
production genre.  The CinematographerAgent injects these constraints into
every Shot before the consistency engine processes them.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-genre forbidden visual elements
# ---------------------------------------------------------------------------

GENRE_NEGATIVES: dict[str, list[str]] = {
    "biblical": [
        "plastic",
        "nylon",
        "modern glass",
        "electronics",
        "watches",
        "sneakers",
        "zippers",
        "sunglasses",
        "asphalt",
        "cars",
        "planes",
        "power lines",
        "street lights",
        "saturated neon dyes",
    ],
    "cyber_noir": [
        "natural bright sunlight",
        "verdant green forests",
        "happy animals",
        "rustic villages",
        "historical accuracy",
        "optimism",
        "flat lighting",
    ],
    "action_kinetic": [
        "motion blur",
        "soft focus",
        "shaky artifacts",
        "static water",
        "slow movement",
        "peaceful atmosphere",
        "serenity",
    ],
}

# Quality guard appended to every negative prompt regardless of genre.
_QUALITY_GUARD = "low quality, distorted anatomy, glitches"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class NegativeConstraintManager:
    """Generates a genre-specific negative prompt for Veo 3.1.

    Args:
        genre: The production genre key (e.g. ``"biblical"``).  An
            unrecognised genre results in an empty forbidden list; only the
            universal quality guard is applied.

    Example::

        mgr = NegativeConstraintManager("biblical")
        shot.negative_prompt = mgr.get_negative_prompt()
    """

    def __init__(self, genre: str) -> None:
        self._genre = genre.lower() if genre else ""
        self.forbidden_list: list[str] = list(
            GENRE_NEGATIVES.get(self._genre, [])
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_negative_prompt(self) -> str:
        """Return the complete negative-prompt string for the active genre.

        The string always ends with the universal quality guard tokens so that
        fundamental render quality is enforced even for unknown genres.

        Returns:
            A comma-separated negative-prompt string ready for injection into
            a Veo 3.1 generation request.
        """
        parts = self.forbidden_list + [_QUALITY_GUARD]
        return ", ".join(parts)
