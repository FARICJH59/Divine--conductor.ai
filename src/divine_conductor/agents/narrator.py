"""NarratorAgent — parses a scripture passage into discrete scenes."""

from __future__ import annotations

import re
from typing import Final

from divine_conductor.agents.base import BaseAgent
from divine_conductor.models.production import (
    EmotionalTone,
    CameraAngle,
    Scene,
    ProductionState,
)

# ---------------------------------------------------------------------------
# Keyword → tone mapping (heuristic, easily extended)
# ---------------------------------------------------------------------------

_TONE_KEYWORDS: Final[dict[EmotionalTone, list[str]]] = {
    EmotionalTone.TRIUMPHANT: ["victory", "triumph", "rejoice", "praise", "glory", "saved"],
    EmotionalTone.SORROWFUL: ["wept", "mourn", "grief", "lament", "sorrow", "tears", "died"],
    EmotionalTone.JOYFUL: ["joy", "glad", "laugh", "celebrate", "sing", "blessed"],
    EmotionalTone.TENSE: ["fear", "battle", "enemy", "sword", "shook", "trembled", "anger"],
    EmotionalTone.PEACEFUL: ["rest", "still", "quiet", "peace", "calm", "gentle"],
    EmotionalTone.AWESTRUCK: ["wonder", "miracle", "appeared", "vision", "holy", "divine"],
    EmotionalTone.REFLECTIVE: ["meditate", "remember", "considered", "pondered", "thought"],
    EmotionalTone.REVERENT: [],  # default fall-through
}

# Sentence boundary pattern — splits on ". ", "! ", "? " or verse markers
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|(?:\d+:\d+\s+)")

# Setting keywords → candidate setting descriptions
_SETTING_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    (re.compile(r"\b(garden|eden)\b", re.I), "lush garden landscape, Eden"),
    (re.compile(r"\b(desert|wilderness)\b", re.I), "vast desert wilderness"),
    (re.compile(r"\b(sea|water|river|jordan)\b", re.I), "shimmering water, reflective surface"),
    (re.compile(r"\b(mountain|mount|sinai|zion|carmel)\b", re.I), "dramatic mountain peak"),
    (re.compile(r"\b(temple|synagogue|tabernacle|sanctuary)\b", re.I), "sacred temple interior"),
    (re.compile(r"\b(city|jerusalem|bethlehem|nazareth)\b", re.I), "ancient stone city streets"),
    (re.compile(r"\b(heaven|sky|cloud|firmament)\b", re.I), "vast celestial sky above"),
    (re.compile(r"\b(cross|calvary|golgotha)\b", re.I), "Golgotha hillside, wooden cross"),
]


class NarratorAgent(BaseAgent):
    """Parses the passage text into a list of ``Scene`` objects.

    Each sentence (or verse) becomes one scene.  Emotional tone and setting
    are inferred via lightweight keyword heuristics so that no external model
    call is required by default.

    Args:
        min_sentence_length: Sentences shorter than this (in characters) are
            merged with the next sentence.
        default_duration: Default ``duration_seconds`` assigned to each scene.
    """

    name = "narrator_agent"

    def __init__(
        self,
        min_sentence_length: int = 20,
        default_duration: float = 5.0,
    ) -> None:
        self._min_length = min_sentence_length
        self._default_duration = default_duration

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, state: ProductionState) -> ProductionState:
        """Split the passage text into scenes and add them to *state*."""
        sentences = self._split_into_sentences(state.config.passage_text)
        scenes: list[Scene] = []
        for idx, sentence in enumerate(sentences):
            scenes.append(
                Scene(
                    index=idx,
                    text=sentence,
                    setting=self._infer_setting(sentence),
                    tone=self._infer_tone(sentence),
                    duration_seconds=self._default_duration,
                    characters=self._find_characters(sentence, state),
                )
            )
        state.scenes = scenes
        state.metadata["narrator_sentence_count"] = len(sentences)
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split *text* into non-trivial sentences."""
        raw = _SENTENCE_RE.split(text.strip())
        cleaned: list[str] = []
        buffer = ""
        for part in raw:
            part = part.strip()
            if not part:
                continue
            buffer = (buffer + " " + part).strip() if buffer else part
            if len(buffer) >= self._min_length:
                cleaned.append(buffer)
                buffer = ""
        if buffer:
            cleaned.append(buffer)
        return cleaned or [text.strip()]

    @staticmethod
    def _infer_tone(text: str) -> EmotionalTone:
        lower = text.lower()
        for tone, keywords in _TONE_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return tone
        return EmotionalTone.REVERENT

    @staticmethod
    def _infer_setting(text: str) -> str:
        for pattern, setting in _SETTING_PATTERNS:
            if pattern.search(text):
                return setting
        return "ancient biblical landscape"

    @staticmethod
    def _find_characters(text: str, state: ProductionState) -> list[str]:
        """Return IDs of registered characters whose names appear in *text*."""
        lower = text.lower()
        found: list[str] = []
        for char in state.config.characters:
            if char.name.lower() in lower or char.id.lower() in lower:
                found.append(char.id)
        return found
