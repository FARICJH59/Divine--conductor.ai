"""AuralAgent — generates audio plans for every shot in the production.

For each shot the agent produces an ``AudioPlan`` that specifies:

* **Ambient_Layer** — a continuous background sound identifier matched to
  the Scene's ``EmotionalTone`` (tone takes priority) or the production's
  visual *style* (genre proxy).
* **Diegetic_Hits** — timed sound-effect cues triggered by the shot's
  *Kinetic Level* (derived from ``CameraAngle``).
* **Musical_Prompt** — a Suno/Udio-compatible natural-language prompt that
  matches the Scene's ``EmotionalTone``.

A ``LoudnessGate`` is applied to every plan to ensure the Narrator's voice
anchors at **−14 LUFS** while environmental layers stay in the background.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from divine_conductor.agents.base import BaseAgent
from divine_conductor.models.production import (
    CameraAngle,
    EmotionalTone,
    ProductionState,
    Shot,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loudness constants
# ---------------------------------------------------------------------------

#: Target integrated loudness for the narrator voice track (LUFS).
NARRATOR_LUFS: float = -14.0

#: Target loudness for continuous ambient background layers (LUFS).
AMBIENT_TARGET_LUFS: float = -24.0

#: Target loudness for diegetic sound-effect hits (LUFS).
DIEGETIC_TARGET_LUFS: float = -20.0


# ---------------------------------------------------------------------------
# Kinetic Level
# ---------------------------------------------------------------------------


class KineticLevel(str, Enum):
    """Derived kinetic intensity of a shot based on its camera angle."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_ANGLE_KINETIC: dict[CameraAngle, KineticLevel] = {
    CameraAngle.WIDE: KineticLevel.LOW,
    CameraAngle.OVERHEAD: KineticLevel.LOW,
    CameraAngle.MEDIUM: KineticLevel.MEDIUM,
    CameraAngle.CLOSE_UP: KineticLevel.MEDIUM,
    CameraAngle.EXTREME_CLOSE_UP: KineticLevel.MEDIUM,
    CameraAngle.LOW_ANGLE: KineticLevel.MEDIUM,
    CameraAngle.POV: KineticLevel.HIGH,
    CameraAngle.DUTCH_ANGLE: KineticLevel.HIGH,
}


def _build_diegetic_hits(
    kinetic: KineticLevel, duration: float
) -> list[dict[str, Any]]:
    """Return a list of timed diegetic sound-effect cues for *kinetic* level."""
    if kinetic == KineticLevel.HIGH:
        return [
            {"time_seconds": 0.0, "effect": "impact_hit", "gain_db": -6.0},
            {
                "time_seconds": round(duration * 0.5, 2),
                "effect": "tension_riser",
                "gain_db": -9.0,
            },
        ]
    if kinetic == KineticLevel.MEDIUM:
        return [
            {"time_seconds": 0.0, "effect": "ambient_transition", "gain_db": -12.0},
        ]
    return []


# ---------------------------------------------------------------------------
# Ambient layer mappings
# ---------------------------------------------------------------------------

# Visual style (genre proxy) → default ambient layer
_STYLE_AMBIENT: dict[str, str] = {
    "cinematic": "orchestral_underscore",
    "documentary": "natural_field_recording",
    "animated": "fantastical_atmospheric_pad",
}

# EmotionalTone → ambient layer (takes priority over style default)
_TONE_AMBIENT: dict[EmotionalTone, str] = {
    EmotionalTone.REVERENT: "sacred_choir_drone",
    EmotionalTone.TRIUMPHANT: "brass_strings_swell",
    EmotionalTone.SORROWFUL: "mournful_strings_pad",
    EmotionalTone.JOYFUL: "warm_acoustic_ambient",
    EmotionalTone.TENSE: "low_frequency_tension_pad",
    EmotionalTone.PEACEFUL: "gentle_wind_nature_ambient",
    EmotionalTone.AWESTRUCK: "ethereal_choir_swell",
    EmotionalTone.REFLECTIVE: "sparse_piano_ambient",
}


# ---------------------------------------------------------------------------
# Suno/Udio musical-prompt mapping
# ---------------------------------------------------------------------------

_TONE_MUSICAL_PROMPT: dict[EmotionalTone, str] = {
    EmotionalTone.REVERENT: (
        "sacred choral composition, slow tempo 60 BPM, "
        "SATB choir with organ, heavy reverb, spiritual hymn style"
    ),
    EmotionalTone.TRIUMPHANT: (
        "epic orchestral fanfare, 120 BPM, full brass section, "
        "snare drum roll, soaring strings, Hans Zimmer inspired"
    ),
    EmotionalTone.SORROWFUL: (
        "sorrowful cinematic score, 50 BPM, solo cello, "
        "sparse piano, minor key, melancholic, grief-laden"
    ),
    EmotionalTone.JOYFUL: (
        "uplifting acoustic score, 110 BPM, acoustic guitar, "
        "light percussion, major key, celebratory, bright"
    ),
    EmotionalTone.TENSE: (
        "suspenseful thriller underscore, 90 BPM, staccato strings, "
        "low brass pulses, dissonant cluster chords, ominous"
    ),
    EmotionalTone.PEACEFUL: (
        "serene ambient music, 70 BPM, soft flute, "
        "gentle strings, nature-inspired, meditative, pastoral"
    ),
    EmotionalTone.AWESTRUCK: (
        "divine awe-inspiring orchestral piece, 80 BPM, "
        "full choir and orchestra, heavenly voices, transcendent, cinematic"
    ),
    EmotionalTone.REFLECTIVE: (
        "contemplative solo piano piece, 65 BPM, "
        "sparse arrangement, introspective, warm, minimal"
    ),
}

_FALLBACK_MUSICAL_PROMPT = (
    "cinematic orchestral underscore, neutral tempo, broad emotional range"
)


# ---------------------------------------------------------------------------
# AudioPlan data structure
# ---------------------------------------------------------------------------


@dataclass
class AudioPlan:
    """Audio design specification for a single shot.

    Attributes:
        shot_id: ID of the corresponding ``Shot``.
        ambient_layer: Identifier for the continuous background sound track.
        diegetic_hits: Ordered list of timed sound-effect cues; each dict
            contains ``time_seconds``, ``effect``, and ``gain_db``.
        musical_prompt: A Suno/Udio-compatible music generation prompt.
        kinetic_level: Derived kinetic intensity (used to gate diegetic hits).
        duration_seconds: Shot duration; used to time diegetic hit offsets.
    """

    shot_id: str
    ambient_layer: str
    diegetic_hits: list[dict[str, Any]] = field(default_factory=list)
    musical_prompt: str = ""
    kinetic_level: KineticLevel = KineticLevel.LOW
    duration_seconds: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "shot_id": self.shot_id,
            "ambient_layer": self.ambient_layer,
            "diegetic_hits": self.diegetic_hits,
            "musical_prompt": self.musical_prompt,
            "kinetic_level": self.kinetic_level.value,
            "duration_seconds": self.duration_seconds,
        }


# ---------------------------------------------------------------------------
# LoudnessGate
# ---------------------------------------------------------------------------


class LoudnessGate:
    """Enforces a loudness hierarchy so the Narrator's voice stays dominant.

    The Narrator track is anchored at ``narrator_lufs`` (default −14 LUFS,
    the standard streaming target).  Ambient and diegetic layers are
    attenuated below that floor so they never mask speech intelligibility.

    Args:
        narrator_lufs: Integrated loudness target for the narrator (LUFS).
        ambient_lufs: Integrated loudness target for ambient layers (LUFS).
        diegetic_lufs: Integrated loudness target for diegetic hits (LUFS).
    """

    def __init__(
        self,
        narrator_lufs: float = NARRATOR_LUFS,
        ambient_lufs: float = AMBIENT_TARGET_LUFS,
        diegetic_lufs: float = DIEGETIC_TARGET_LUFS,
    ) -> None:
        if narrator_lufs <= ambient_lufs:
            raise ValueError(
                "narrator_lufs must be louder than ambient_lufs "
                f"({narrator_lufs} <= {ambient_lufs})."
            )
        if narrator_lufs <= diegetic_lufs:
            raise ValueError(
                "narrator_lufs must be louder than diegetic_lufs "
                f"({narrator_lufs} <= {diegetic_lufs})."
            )
        self.narrator_lufs = narrator_lufs
        self.ambient_lufs = ambient_lufs
        self.diegetic_lufs = diegetic_lufs

    def narrator_headroom_db(self) -> float:
        """Return loudness headroom (dB) between narrator and ambient layer."""
        return self.narrator_lufs - self.ambient_lufs

    def gate_audio_plan(self, plan: AudioPlan) -> dict[str, Any]:
        """Compute mixing gain directives for *plan*.

        Returns a dict with relative gain adjustments (in dB) that position
        the narrator voice clearly above all environmental audio layers.
        """
        ambient_gain_db = round(self.ambient_lufs - self.narrator_lufs, 1)
        diegetic_gain_db = round(self.diegetic_lufs - self.narrator_lufs, 1)
        return {
            "shot_id": plan.shot_id,
            "narrator_lufs": self.narrator_lufs,
            "ambient_layer_gain_db": ambient_gain_db,
            "diegetic_gain_db": diegetic_gain_db,
            "mix_note": (
                f"Narrator at {self.narrator_lufs} LUFS; "
                f"ambient ducked {abs(ambient_gain_db):.0f} dB; "
                f"diegetic ducked {abs(diegetic_gain_db):.0f} dB"
            ),
        }


# ---------------------------------------------------------------------------
# AuralAgent
# ---------------------------------------------------------------------------


class AuralAgent(BaseAgent):
    """Generates an ``AudioPlan`` for every shot in the production.

    This agent must run **after** the ``CinematographerAgent`` (so that
    ``Shot.duration_seconds`` and ``Shot.camera_angle`` are finalised) and
    **before** final assembly so audio IDs can be included in the manifest.

    Results are stored in ``state.metadata``:

    * ``"audio_plans"`` — list of ``AudioPlan.to_dict()`` dicts, one per shot.
    * ``"loudness_directives"`` — list of mixing gain dicts from
      ``LoudnessGate.gate_audio_plan()``.
    * ``"aural_narrator_lufs"`` — the narrator loudness target applied.

    Args:
        loudness_gate: Optional custom ``LoudnessGate``; defaults to the
            broadcast standard (−14 LUFS narrator, −24 LUFS ambient,
            −20 LUFS diegetic).
    """

    name = "aural_agent"

    def __init__(self, loudness_gate: LoudnessGate | None = None) -> None:
        self._gate = loudness_gate or LoudnessGate()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, state: ProductionState) -> ProductionState:
        """Generate ``AudioPlan`` objects for all shots in *state*."""
        scene_tone: dict[str, EmotionalTone] = {
            scene.id: scene.tone for scene in state.scenes
        }
        style = state.config.style

        audio_plans: list[AudioPlan] = []
        for shot in state.shots:
            tone = scene_tone.get(shot.scene_id, EmotionalTone.REVERENT)
            audio_plans.append(self._build_audio_plan(shot, tone, style))

        loudness_directives = [
            self._gate.gate_audio_plan(plan) for plan in audio_plans
        ]

        state.metadata["audio_plans"] = [p.to_dict() for p in audio_plans]
        state.metadata["loudness_directives"] = loudness_directives
        state.metadata["aural_narrator_lufs"] = self._gate.narrator_lufs

        logger.info(
            "[%s] Generated %d audio plans (narrator target: %.1f LUFS).",
            self.name,
            len(audio_plans),
            self._gate.narrator_lufs,
        )
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_audio_plan(
        shot: Shot,
        tone: EmotionalTone,
        style: str,
    ) -> AudioPlan:
        """Compose an ``AudioPlan`` for a single *shot*."""
        kinetic = _ANGLE_KINETIC.get(shot.camera_angle, KineticLevel.MEDIUM)

        # Tone-specific ambient takes priority over style/genre default
        ambient_layer = _TONE_AMBIENT.get(
            tone,
            _STYLE_AMBIENT.get(style, "orchestral_underscore"),
        )

        diegetic_hits = _build_diegetic_hits(kinetic, shot.duration_seconds)

        musical_prompt = _TONE_MUSICAL_PROMPT.get(tone, _FALLBACK_MUSICAL_PROMPT)

        return AudioPlan(
            shot_id=shot.id,
            ambient_layer=ambient_layer,
            diegetic_hits=diegetic_hits,
            musical_prompt=musical_prompt,
            kinetic_level=kinetic,
            duration_seconds=shot.duration_seconds,
        )
