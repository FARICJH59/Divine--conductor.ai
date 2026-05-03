"""Core data models for the Divine Conductor production pipeline.

Defines the immutable value objects and mutable state containers used
throughout the agentic pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CameraAngle(str, Enum):
    """Standard cinematic camera angles used by the DirectorAgent."""

    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    OVERHEAD = "overhead"
    LOW_ANGLE = "low_angle"
    POV = "pov"
    DUTCH_ANGLE = "dutch_angle"


class EmotionalTone(str, Enum):
    """High-level emotional tone for a scene."""

    REVERENT = "reverent"
    TRIUMPHANT = "triumphant"
    SORROWFUL = "sorrowful"
    JOYFUL = "joyful"
    TENSE = "tense"
    PEACEFUL = "peaceful"
    AWESTRUCK = "awestruck"
    REFLECTIVE = "reflective"


class PalettePreset(str, Enum):
    """Named colour-grade presets understood by the Veo 3.1 consistency engine."""

    WARM_GOLDEN_DAWN = "warm_golden_dawn"
    DESERT_NOON = "desert_noon"
    TWILIGHT_SACRED = "twilight_sacred"
    MOONLIT_BLUE = "moonlit_blue"
    STORM_GREY = "storm_grey"
    DIVINE_WHITE = "divine_white"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Character:
    """A character that appears in the production.

    Attributes:
        id: Unique stable identifier used to anchor visual consistency.
        name: Display name (may be used in prompts).
        description: Canonical visual description injected into shot prompts.
        role: Narrative role (e.g. "protagonist", "prophet", "narrator").
    """

    id: str
    name: str
    description: str
    role: str = "supporting"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Character.id must not be empty.")
        if not self.description:
            raise ValueError("Character.description must not be empty.")


@dataclass
class Scene:
    """A discrete narrative unit produced by the NarratorAgent.

    Attributes:
        id: Auto-generated unique identifier.
        index: Zero-based position in the production sequence.
        text: Raw scriptural or screenplay text for this scene.
        setting: Brief description of the physical location/time.
        characters: List of character IDs present in the scene.
        tone: Dominant emotional tone.
        director_notes: Annotations added by the DirectorAgent.
        camera_angle: Suggested primary camera angle (set by DirectorAgent).
        duration_seconds: Estimated screen time in seconds.
    """

    index: int
    text: str
    setting: str = ""
    characters: list[str] = field(default_factory=list)
    tone: EmotionalTone = EmotionalTone.REVERENT
    director_notes: str = ""
    camera_angle: CameraAngle = CameraAngle.WIDE
    duration_seconds: float = 5.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("Scene.text must not be empty.")
        if self.duration_seconds <= 0:
            raise ValueError("Scene.duration_seconds must be positive.")


@dataclass
class StyleConflictMetadata:
    """Metadata describing a detected style conflict between consecutive shots.

    Used by the ``TemporalPacingAgent`` to modulate timing and motion
    parameters and inject corrective flow anchors.

    Attributes:
        conflict_type: Category of conflict (e.g. ``"motion_vector"``,
            ``"temporal"``, ``"tonal"``).
        severity: Conflict intensity on a 0.0–1.0 scale (0 = minor, 1 = severe).
        motion_bucket: Suggested video-generation motion intensity (1–255).
            Higher values produce more motion.
        description: Human-readable summary of the conflict.
    """

    conflict_type: str
    severity: float = 0.0
    motion_bucket: int = 127
    description: str = ""

    def __post_init__(self) -> None:
        if not self.conflict_type:
            raise ValueError("StyleConflictMetadata.conflict_type must not be empty.")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("StyleConflictMetadata.severity must be between 0 and 1.")
        if not 1 <= self.motion_bucket <= 255:
            raise ValueError("StyleConflictMetadata.motion_bucket must be between 1 and 255.")


@dataclass
class Shot:
    """A single camera shot produced by the CinematographerAgent.

    This is the terminal output of the pipeline — each Shot maps 1-to-1 with
    a Veo 3.1 generation request.

    Attributes:
        scene_id: ID of the parent Scene.
        index: Zero-based shot index within the scene.
        prompt: Full Veo-compatible text prompt.
        negative_prompt: Optional negative-prompt hints.
        duration_seconds: Target clip length.
        camera_angle: Camera angle for this shot.
        motion_bucket: Video-generation motion intensity (1–255).
        consistency_anchors: Key/value pairs injected by the consistency engine.
        id: Auto-generated unique identifier.
    """

    scene_id: str
    index: int
    prompt: str
    negative_prompt: str = ""
    duration_seconds: float = 5.0
    camera_angle: CameraAngle = CameraAngle.WIDE
    motion_bucket: int = 127
    consistency_anchors: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("Shot.prompt must not be empty.")
        if not 1 <= self.motion_bucket <= 255:
            raise ValueError("Shot.motion_bucket must be between 1 and 255.")


# ---------------------------------------------------------------------------
# Configuration & pipeline state
# ---------------------------------------------------------------------------


@dataclass
class ProductionConfig:
    """Top-level pipeline configuration.

    Attributes:
        name: Human-readable production title.
        passage_text: The scriptural or screenplay source text.
        style: Visual style preset (cinematic | documentary | animated).
        aspect_ratio: Target aspect ratio string (e.g. "16:9").
        fps: Target frame rate.
        palette: Colour palette preset.
        character_id_strength: How strongly character anchors are applied (0–1).
        anchor_shots: Whether to generate anchor reference shots first.
        output_format: Serialisation format for the final shot bundle.
        output_path: Directory path for output artefacts.
        characters: Pre-defined characters to register in the consistency engine.
    """

    name: str
    passage_text: str
    style: str = "cinematic"
    aspect_ratio: str = "16:9"
    fps: int = 24
    palette: PalettePreset = PalettePreset.WARM_GOLDEN_DAWN
    character_id_strength: float = 0.85
    anchor_shots: bool = True
    output_format: str = "json"
    output_path: str = "output"
    characters: list[Character] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ProductionConfig.name must not be empty.")
        if not self.passage_text:
            raise ValueError("ProductionConfig.passage_text must not be empty.")
        if not 0.0 <= self.character_id_strength <= 1.0:
            raise ValueError("character_id_strength must be between 0 and 1.")
        if self.style not in {"cinematic", "documentary", "animated"}:
            raise ValueError(
                f"Unsupported style '{self.style}'. "
                "Choose from: cinematic, documentary, animated."
            )


@dataclass
class ProductionState:
    """Mutable container that flows through the entire agentic pipeline.

    Each agent reads from and writes to this object, progressively enriching
    the production data from raw text to a fully-specified shot list.

    Attributes:
        config: Immutable production configuration.
        scenes: Scenes built by NarratorAgent, annotated by DirectorAgent.
        shots: Final shot list built by CinematographerAgent.
        consistency_report: Issues flagged by the Veo3ConsistencyEngine.
        metadata: Arbitrary key/value metadata from agents.
    """

    config: ProductionConfig
    scenes: list[Scene] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    consistency_report: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def scenes_for_character(self, character_id: str) -> list[Scene]:
        """Return all scenes that include the given character ID."""
        return [s for s in self.scenes if character_id in s.characters]

    def shots_for_scene(self, scene_id: str) -> list[Shot]:
        """Return all shots that belong to the given scene."""
        return [sh for sh in self.shots if sh.scene_id == scene_id]

    @property
    def total_duration_seconds(self) -> float:
        """Estimated total production duration based on shot durations."""
        return sum(sh.duration_seconds for sh in self.shots)

    def summary(self) -> dict[str, Any]:
        """Return a concise summary dict suitable for logging."""
        return {
            "production": self.config.name,
            "style": self.config.style,
            "scenes": len(self.scenes),
            "shots": len(self.shots),
            "total_duration_seconds": self.total_duration_seconds,
            "consistency_issues": len(self.consistency_report),
        }
