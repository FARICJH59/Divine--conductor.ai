"""StageManagerAgent — assigns 3-D spatial layout to every scene.

The agent builds a :class:`StageMap` for each scene by mapping the scene's
``camera_angle`` to a camera-world position and its ``tone`` to a key-light
direction vector.  The resulting natural-language orientation tokens are
written back to ``scene.spatial_tokens`` so the ``CinematographerAgent``
can inject them directly into Veo-compatible shot prompts.

All spatial maps are stored in ``state.metadata["stage_maps"]`` as a
``dict[scene_id, StageMap]`` for downstream consumers such as the
``BatchSequenceController``.
"""

from __future__ import annotations

from divine_conductor.agents.base import BaseAgent
from divine_conductor.models.production import (
    CameraAngle,
    Coords3D,
    EmotionalTone,
    ProductionState,
    Scene,
    StageMap,
)

# ---------------------------------------------------------------------------
# Camera-angle → world-space camera position
#
# Convention: primary subject is fixed at origin (0, 0, 0).
#   x: negative = camera left,  positive = camera right
#   y: negative = camera below, positive = camera above
#   z: positive = camera in front of subject (standard perspective)
# ---------------------------------------------------------------------------

_ANGLE_CAMERA_POS: dict[CameraAngle, Coords3D] = {
    CameraAngle.WIDE:              Coords3D(x=0.0,  y=0.5,  z=8.0),
    CameraAngle.MEDIUM:            Coords3D(x=0.0,  y=0.0,  z=4.0),
    CameraAngle.CLOSE_UP:          Coords3D(x=0.2,  y=0.0,  z=2.0),
    CameraAngle.EXTREME_CLOSE_UP:  Coords3D(x=0.0,  y=0.0,  z=0.8),
    CameraAngle.OVERHEAD:          Coords3D(x=0.0,  y=6.0,  z=0.5),
    CameraAngle.LOW_ANGLE:         Coords3D(x=0.0,  y=-2.0, z=3.0),
    CameraAngle.POV:               Coords3D(x=0.1,  y=0.0,  z=1.5),
    CameraAngle.DUTCH_ANGLE:       Coords3D(x=-0.5, y=0.3,  z=3.5),
}

# ---------------------------------------------------------------------------
# Tone → key-light motivation direction vector
#
# The vector points FROM the light source TOWARD the subject.
#   Positive x → light from the right
#   Positive y → light from above
#   Positive z → light from the front
# ---------------------------------------------------------------------------

_TONE_KEY_LIGHT: dict[EmotionalTone, Coords3D] = {
    EmotionalTone.REVERENT:   Coords3D(x=0.3,  y=1.0,  z=0.5),
    EmotionalTone.TRIUMPHANT: Coords3D(x=1.0,  y=0.5,  z=-1.0),
    EmotionalTone.SORROWFUL:  Coords3D(x=-1.0, y=0.0,  z=1.0),
    EmotionalTone.JOYFUL:     Coords3D(x=0.5,  y=1.0,  z=1.0),
    EmotionalTone.TENSE:      Coords3D(x=-0.8, y=-0.5, z=1.0),
    EmotionalTone.PEACEFUL:   Coords3D(x=0.5,  y=0.8,  z=0.5),
    EmotionalTone.AWESTRUCK:  Coords3D(x=0.0,  y=2.0,  z=0.0),
    EmotionalTone.REFLECTIVE: Coords3D(x=-0.5, y=0.3,  z=1.0),
}

# Fallback positions used when a key is not in the lookup tables
_DEFAULT_CAMERA_POS = Coords3D(x=0.0, y=0.0, z=5.0)
_DEFAULT_KEY_LIGHT   = Coords3D(x=0.5, y=0.5, z=0.5)

# Primary subject is always placed at the world origin
_SUBJECT_ORIGIN = Coords3D(x=0.0, y=0.0, z=0.0)


class StageManagerAgent(BaseAgent):
    """Assigns a :class:`StageMap` to every scene in the production.

    The agent runs **before** the ``CinematographerAgent`` so that
    ``scene.spatial_tokens`` are available when shot prompts are built.

    After the run:
    * Each ``scene.spatial_tokens`` contains a comma-separated string of
      cinematic framing descriptors (e.g. ``"camera centered on subject,
      eye-level framing, wide establishing distance, key light motivated
      from above"``).
    * ``state.metadata["stage_maps"]`` is a ``dict[str, StageMap]``
      mapping every scene's ID to its spatial layout.
    """

    name = "stage_manager_agent"

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, state: ProductionState) -> ProductionState:
        """Build stage maps for every scene and inject spatial tokens."""
        stage_maps: dict[str, StageMap] = {}
        for scene in state.scenes:
            stage_map = self._build_stage_map(scene)
            scene.spatial_tokens = stage_map.get_spatial_prompt()
            stage_maps[scene.id] = stage_map
        state.metadata["stage_maps"] = stage_maps
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_stage_map(self, scene: Scene) -> StageMap:
        """Derive a :class:`StageMap` for *scene* from its angle and tone."""
        camera_pos = _ANGLE_CAMERA_POS.get(scene.camera_angle, _DEFAULT_CAMERA_POS)
        key_light  = _TONE_KEY_LIGHT.get(scene.tone, _DEFAULT_KEY_LIGHT)
        return StageMap(
            scene_id=scene.id,
            primary_subject=_SUBJECT_ORIGIN,
            camera_position=camera_pos,
            key_light_motivation=key_light,
        )
