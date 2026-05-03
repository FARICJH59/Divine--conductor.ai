"""Veo 3.1 consistency engine.

Maintains visual coherence across all shots in a production by:
  - anchoring character appearances to canonical descriptions
  - locking a colour palette across every prompt
  - detecting continuity contradictions between consecutive shots
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from divine_conductor.models.production import (
    Character,
    PalettePreset,
    Shot,
)


# ---------------------------------------------------------------------------
# Palette definitions
# ---------------------------------------------------------------------------

_PALETTE_DESCRIPTIONS: dict[PalettePreset, str] = {
    PalettePreset.WARM_GOLDEN_DAWN: (
        "warm golden-hour lighting, amber and ochre tones, soft diffused sunlight, "
        "long gentle shadows, cinematic lens flare"
    ),
    PalettePreset.DESERT_NOON: (
        "harsh midday desert sun, bleached sand tones, high contrast, deep blue sky, "
        "shimmering heat haze"
    ),
    PalettePreset.TWILIGHT_SACRED: (
        "sacred blue-hour twilight, deep indigo and violet sky, candlelight warmth, "
        "mystical atmospheric haze"
    ),
    PalettePreset.MOONLIT_BLUE: (
        "cool moonlit night, silver-blue luminance, deep shadows, starfield backdrop, "
        "ethereal glow"
    ),
    PalettePreset.STORM_GREY: (
        "overcast storm light, dramatic grey-green sky, high contrast shadows, "
        "ominous cloud formations, desaturated palette"
    ),
    PalettePreset.DIVINE_WHITE: (
        "brilliant divine white light, soft overexposed glow, angelic luminosity, "
        "minimal shadow, transcendent atmosphere"
    ),
}

# Patterns used by the continuity checker to detect contradictions
_TIME_OF_DAY_TOKENS: list[str] = [
    "dawn", "sunrise", "morning", "noon", "midday", "afternoon",
    "dusk", "sunset", "twilight", "evening", "night", "midnight",
]

_WEATHER_TOKENS: list[str] = [
    "sunny", "clear sky", "cloudy", "overcast", "rain", "storm",
    "snow", "fog", "mist", "wind",
]

# Maps wardrobe interaction-rule keys to the prompt keywords that trigger them
_WARDROBE_ENV_KEYWORDS: dict[str, list[str]] = {
    "in_water": ["water", "river", "lake", "sea", "ocean", "rain", "flood", "pool", "stream", "bath"],
    "in_sunlight": ["sunlight", "sunshine", "sun", "bright", "daylight", "noon", "midday"],
    "in_shadow": ["shadow", "shade", "darkness", "dark", "silhouette"],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharacterAnchor:
    """Canonical visual anchor for a single character.

    Attributes:
        character_id: Matches ``Character.id``.
        anchor_text: The visual description injected into prompts.
        strength: Weight applied when merging with the shot's own description (0–1).
        wardrobe_rules: Mapping of environment keys (e.g. ``"in_water"``) to
            rendering-hint strings injected when matching keywords are found
            in a shot prompt.
    """

    character_id: str
    anchor_text: str
    strength: float = 0.85
    wardrobe_rules: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("CharacterAnchor.strength must be between 0 and 1.")


@dataclass(frozen=True)
class PaletteAnchor:
    """Locked colour grade applied to every shot in the production.

    Attributes:
        preset: The ``PalettePreset`` enum value.
        description: Human-readable prompt fragment injected into each shot.
    """

    preset: PalettePreset
    description: str


@dataclass
class ContinuityIssue:
    """A detected continuity contradiction between two consecutive shots.

    Attributes:
        shot_a_id: ID of the earlier shot.
        shot_b_id: ID of the later shot.
        category: Issue type (e.g. "time_of_day", "weather").
        detail: Human-readable description of the contradiction.
    """

    shot_a_id: str
    shot_b_id: str
    category: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_a_id": self.shot_a_id,
            "shot_b_id": self.shot_b_id,
            "category": self.category,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


class Veo3ConsistencyEngine:
    """Ensures cross-shot visual coherence for a Veo 3.1 production.

    Usage::

        engine = Veo3ConsistencyEngine(
            palette=PalettePreset.WARM_GOLDEN_DAWN,
            character_id_strength=0.85,
        )
        engine.register_character(my_character)
        enhanced_shots = engine.apply(shots)
        issues = engine.check_continuity(enhanced_shots)
    """

    def __init__(
        self,
        palette: PalettePreset = PalettePreset.WARM_GOLDEN_DAWN,
        character_id_strength: float = 0.85,
    ) -> None:
        self._palette_anchor = PaletteAnchor(
            preset=palette,
            description=_PALETTE_DESCRIPTIONS[palette],
        )
        self._character_id_strength = character_id_strength
        self._character_anchors: dict[str, CharacterAnchor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_character(self, character: Character) -> None:
        """Register a character so its visual anchor is injected into prompts."""
        wardrobe_rules: dict[str, str] = dict(
            character.wardrobe_logic.get("interaction_rules", {})
        )
        anchor = CharacterAnchor(
            character_id=character.id,
            anchor_text=character.description,
            strength=self._character_id_strength,
            wardrobe_rules=wardrobe_rules,
        )
        self._character_anchors[character.id] = anchor

    def apply(self, shots: list[Shot]) -> list[Shot]:
        """Return a new list of shots with consistency anchors applied.

        Each returned ``Shot`` has its ``prompt`` enriched with palette and
        character anchor text, and its ``consistency_anchors`` dict populated.
        The original shots are **not** mutated.
        """
        enriched: list[Shot] = []
        for shot in shots:
            enriched.append(self._enrich_shot(shot))
        return enriched

    def check_continuity(self, shots: list[Shot]) -> list[ContinuityIssue]:
        """Scan consecutive shot pairs for continuity contradictions.

        Currently checks:
        - Time-of-day jumps (e.g. "sunrise" → "midnight" without transition)
        - Weather contradictions (e.g. "clear sky" → "storm")

        Returns a (possibly empty) list of ``ContinuityIssue`` objects.
        """
        issues: list[ContinuityIssue] = []
        for i in range(len(shots) - 1):
            a, b = shots[i], shots[i + 1]
            issues.extend(self._check_time_of_day(a, b))
            issues.extend(self._check_weather(a, b))
        return issues

    @property
    def palette_anchor(self) -> PaletteAnchor:
        return self._palette_anchor

    @property
    def character_anchors(self) -> dict[str, CharacterAnchor]:
        return dict(self._character_anchors)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enrich_shot(self, shot: Shot) -> Shot:
        """Return an enriched copy of *shot*."""
        anchors: dict[str, Any] = dict(shot.consistency_anchors)
        prompt_parts: list[str] = [shot.prompt]

        # Inject palette
        palette_text = self._palette_anchor.description
        anchors["palette"] = palette_text
        prompt_parts.append(palette_text)

        # Inject character anchors that appear in the prompt
        for char_id, anchor in self._character_anchors.items():
            if char_id.lower() in shot.prompt.lower():
                anchors[f"character:{char_id}"] = anchor.anchor_text
                prompt_parts.append(anchor.anchor_text)
                # Inject context-sensitive wardrobe rendering hints
                for env_key, rule_text in anchor.wardrobe_rules.items():
                    keywords = _WARDROBE_ENV_KEYWORDS.get(env_key)
                    if keywords is None:
                        # Unknown environment key — skip rather than guess
                        continue
                    if any(kw in shot.prompt.lower() for kw in keywords):
                        anchors[f"wardrobe:{char_id}:{env_key}"] = rule_text
                        prompt_parts.append(rule_text)

        # Inject Veo 3.1 quality tokens
        anchors["veo_model"] = "veo-3.1"
        prompt_parts.append(
            "photorealistic cinematic quality, 8K, shot on ARRI ALEXA 35, "
            "high dynamic range, film grain"
        )

        enriched_prompt = ", ".join(prompt_parts)

        return Shot(
            scene_id=shot.scene_id,
            index=shot.index,
            prompt=enriched_prompt,
            negative_prompt=shot.negative_prompt,
            duration_seconds=shot.duration_seconds,
            camera_angle=shot.camera_angle,
            consistency_anchors=anchors,
            id=shot.id,
        )

    @staticmethod
    def _extract_tokens(text: str, token_list: list[str]) -> set[str]:
        """Return the subset of *token_list* that appear in *text*."""
        lower = text.lower()
        return {tok for tok in token_list if re.search(r"\b" + re.escape(tok) + r"\b", lower)}

    def _check_time_of_day(self, a: Shot, b: Shot) -> list[ContinuityIssue]:
        tokens_a = self._extract_tokens(a.prompt, _TIME_OF_DAY_TOKENS)
        tokens_b = self._extract_tokens(b.prompt, _TIME_OF_DAY_TOKENS)
        if tokens_a and tokens_b and not tokens_a.intersection(tokens_b):
            return [
                ContinuityIssue(
                    shot_a_id=a.id,
                    shot_b_id=b.id,
                    category="time_of_day",
                    detail=(
                        f"Shot {a.id[:8]} references {tokens_a} "
                        f"but shot {b.id[:8]} references {tokens_b}. "
                        "Verify the temporal transition is intentional."
                    ),
                )
            ]
        return []

    def _check_weather(self, a: Shot, b: Shot) -> list[ContinuityIssue]:
        tokens_a = self._extract_tokens(a.prompt, _WEATHER_TOKENS)
        tokens_b = self._extract_tokens(b.prompt, _WEATHER_TOKENS)
        if tokens_a and tokens_b and not tokens_a.intersection(tokens_b):
            return [
                ContinuityIssue(
                    shot_a_id=a.id,
                    shot_b_id=b.id,
                    category="weather",
                    detail=(
                        f"Shot {a.id[:8]} references {tokens_a} "
                        f"but shot {b.id[:8]} references {tokens_b}. "
                        "Verify the weather change is intentional."
                    ),
                )
            ]
        return []
