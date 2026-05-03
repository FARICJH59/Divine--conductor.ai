"""DirectorAgent — annotates scenes with camera work and pacing."""

from __future__ import annotations

from divine_conductor.agents.base import BaseAgent
from divine_conductor.models.production import (
    CameraAngle,
    EmotionalTone,
    ProductionState,
    Scene,
)

# ---------------------------------------------------------------------------
# Tone → default camera angle heuristic
# ---------------------------------------------------------------------------

_TONE_TO_ANGLE: dict[EmotionalTone, CameraAngle] = {
    EmotionalTone.AWESTRUCK: CameraAngle.LOW_ANGLE,
    EmotionalTone.TRIUMPHANT: CameraAngle.WIDE,
    EmotionalTone.SORROWFUL: CameraAngle.CLOSE_UP,
    EmotionalTone.JOYFUL: CameraAngle.MEDIUM,
    EmotionalTone.TENSE: CameraAngle.DUTCH_ANGLE,
    EmotionalTone.PEACEFUL: CameraAngle.OVERHEAD,
    EmotionalTone.REFLECTIVE: CameraAngle.MEDIUM,
    EmotionalTone.REVERENT: CameraAngle.LOW_ANGLE,
}

# ---------------------------------------------------------------------------
# Style-specific pacing multipliers
# ---------------------------------------------------------------------------

_STYLE_PACE: dict[str, float] = {
    "cinematic": 1.0,
    "documentary": 1.4,
    "animated": 0.75,
}

# ---------------------------------------------------------------------------
# Director's note templates
# ---------------------------------------------------------------------------

_DIRECTOR_NOTE_TEMPLATES: dict[EmotionalTone, str] = {
    EmotionalTone.REVERENT: (
        "Hold on subject with gravitas. Minimal camera movement. "
        "Allow silence to build spiritual weight."
    ),
    EmotionalTone.TRIUMPHANT: (
        "Push in with energy. Swell score cue. "
        "Reveal the wider landscape on the climactic beat."
    ),
    EmotionalTone.SORROWFUL: (
        "Stay tight on the face. Let grief breathe. "
        "Handheld to convey raw emotion."
    ),
    EmotionalTone.JOYFUL: (
        "Wide and bright. Movement in frame. "
        "Score lifts. Consider slow-motion for peak moment."
    ),
    EmotionalTone.TENSE: (
        "Quick cuts. Dutch angle. "
        "Underscore with low rumble. Keep audience off-balance."
    ),
    EmotionalTone.PEACEFUL: (
        "Drone-style overhead glide. "
        "Gentle ambient score. Let nature fill the frame."
    ),
    EmotionalTone.AWESTRUCK: (
        "Low angle looking up. Lens flare on light source. "
        "Slow reveal — hold the moment of recognition."
    ),
    EmotionalTone.REFLECTIVE: (
        "Medium shot, slightly soft focus on background. "
        "Quiet internal score. Long take."
    ),
}


class DirectorAgent(BaseAgent):
    """Annotates each ``Scene`` with camera work, pacing, and director's notes.

    The DirectorAgent reads the tone inferred by the NarratorAgent and applies
    a set of cinematic rules to assign:
      - ``camera_angle``
      - ``duration_seconds`` (scaled by production style)
      - ``director_notes``

    Args:
        base_shot_duration: Default scene duration before style scaling.
    """

    name = "director_agent"

    def __init__(self, base_shot_duration: float = 5.0) -> None:
        self._base_duration = base_shot_duration

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, state: ProductionState) -> ProductionState:
        """Annotate all scenes in *state* with directorial guidance."""
        pace_multiplier = _STYLE_PACE.get(state.config.style, 1.0)
        for scene in state.scenes:
            self._direct_scene(scene, pace_multiplier)
        state.metadata["director_style"] = state.config.style
        state.metadata["director_pace_multiplier"] = pace_multiplier
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _direct_scene(self, scene: Scene, pace_multiplier: float) -> None:
        """Mutate *scene* with camera and pacing annotations."""
        scene.camera_angle = _TONE_TO_ANGLE.get(scene.tone, CameraAngle.WIDE)
        scene.duration_seconds = round(self._base_duration * pace_multiplier, 2)
        scene.director_notes = _DIRECTOR_NOTE_TEMPLATES.get(
            scene.tone, "Standard coverage."
        )
