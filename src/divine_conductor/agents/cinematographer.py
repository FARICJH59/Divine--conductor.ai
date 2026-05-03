"""CinematographerAgent — converts scenes into Veo-compatible shot prompts."""

from __future__ import annotations

from divine_conductor.agents.base import BaseAgent
from divine_conductor.models.production import (
    CameraAngle,
    EmotionalTone,
    ProductionState,
    Scene,
    Shot,
)

# ---------------------------------------------------------------------------
# Camera angle → prompt fragment
# ---------------------------------------------------------------------------

_ANGLE_PROMPTS: dict[CameraAngle, str] = {
    CameraAngle.WIDE: "epic wide shot, vast landscape in frame",
    CameraAngle.MEDIUM: "medium shot, subject centred, natural depth-of-field",
    CameraAngle.CLOSE_UP: "intimate close-up, emotion visible on face",
    CameraAngle.EXTREME_CLOSE_UP: "extreme close-up, eyes in sharp detail",
    CameraAngle.OVERHEAD: "bird's-eye overhead perspective, looking straight down",
    CameraAngle.LOW_ANGLE: "low-angle shot looking upward, subject looms large",
    CameraAngle.POV: "first-person point-of-view, immersive perspective",
    CameraAngle.DUTCH_ANGLE: "Dutch angle, camera tilted 15 degrees, tension in composition",
}

# ---------------------------------------------------------------------------
# Tone → atmospheric lighting fragment
# ---------------------------------------------------------------------------

_TONE_LIGHTING: dict[EmotionalTone, str] = {
    EmotionalTone.REVERENT: "soft volumetric god rays, sacred atmosphere",
    EmotionalTone.TRIUMPHANT: "golden backlighting, flare on hero, triumphant glow",
    EmotionalTone.SORROWFUL: "cold desaturated side-light, deep shadows, grey undertones",
    EmotionalTone.JOYFUL: "bright natural daylight, warm fill, joyful chromatic energy",
    EmotionalTone.TENSE: "harsh rim lighting, underexposed midtones, ominous shadows",
    EmotionalTone.PEACEFUL: "soft diffused overcast light, gentle pastel tones",
    EmotionalTone.AWESTRUCK: "blinding divine light source, lens flare, overexposed highlights",
    EmotionalTone.REFLECTIVE: "soft window light, muted warm tones, contemplative stillness",
}

# ---------------------------------------------------------------------------
# Style → render quality suffix
# ---------------------------------------------------------------------------

_STYLE_QUALITY: dict[str, str] = {
    "cinematic": "anamorphic lens, film grain, cinematic colour grade",
    "documentary": "handheld verité style, natural light, observational framing",
    "animated": "painterly animation style, vivid colours, stylised brushwork",
}


class CinematographerAgent(BaseAgent):
    """Builds ``Shot`` objects from annotated ``Scene`` objects.

    Each scene produces exactly one primary shot.  The prompt is assembled
    from the scene text, director's notes, camera angle, tone-based lighting,
    and style quality hints.  The consistency engine then further enriches
    these prompts downstream.

    Args:
        shots_per_scene: Number of shots to generate per scene (default 1).
    """

    name = "cinematographer_agent"

    def __init__(self, shots_per_scene: int = 1) -> None:
        if shots_per_scene < 1:
            raise ValueError("shots_per_scene must be >= 1.")
        self._shots_per_scene = shots_per_scene

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, state: ProductionState) -> ProductionState:
        """Generate shots for every scene in *state*."""
        shots: list[Shot] = []
        quality_suffix = _STYLE_QUALITY.get(state.config.style, "")
        for scene in state.scenes:
            for shot_idx in range(self._shots_per_scene):
                shots.append(self._build_shot(scene, shot_idx, quality_suffix))
        state.shots = shots
        state.metadata["cinematographer_shots_per_scene"] = self._shots_per_scene
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_shot(self, scene: Scene, shot_idx: int, quality_suffix: str) -> Shot:
        """Compose a full Veo-compatible prompt for *scene*."""
        parts: list[str] = []

        # Core narrative content
        parts.append(scene.text)

        # Setting context
        if scene.setting:
            parts.append(scene.setting)

        # Camera composition
        parts.append(_ANGLE_PROMPTS.get(scene.camera_angle, ""))

        # Tone-driven lighting
        parts.append(_TONE_LIGHTING.get(scene.tone, ""))

        # Director's intent (trimmed to avoid redundancy)
        if scene.director_notes:
            parts.append(scene.director_notes)

        # Style quality
        if quality_suffix:
            parts.append(quality_suffix)

        prompt = ", ".join(p for p in parts if p)

        return Shot(
            scene_id=scene.id,
            index=shot_idx,
            prompt=prompt,
            duration_seconds=scene.duration_seconds,
            camera_angle=scene.camera_angle,
        )
