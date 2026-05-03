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
class Genre:
    """Visual DNA for a named production genre.

    A genre bundles the recurring visual/technical elements that define its
    look and feel.  The ``Veo3ConsistencyEngine`` injects these into every
    shot prompt when a genre is active.

    Attributes:
        key: Machine-readable identifier (e.g. ``"cyber_noir"``).
        visual_anchors: List of recurring visual motifs injected verbatim into
            every shot (e.g. ``["neon reflections", "digital rain"]``).
        lighting: Global lighting description applied to all shots.
        camera_tech: Camera and film specification that overrides the default
            Veo quality-token suffix when set.
        wardrobe_modifier: Genre-wide wardrobe hint appended to every shot.
    """

    key: str
    visual_anchors: list[str] = field(default_factory=list)
    lighting: str = ""
    camera_tech: str = ""
    wardrobe_modifier: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Genre.key must not be empty.")


@dataclass(frozen=True)
class StyleConflictMetadata:
    """Kinetic override parameters that impose an action-genre lens on a scene.

    When active, the ``DirectorAgent`` calls
    :meth:`~divine_conductor.agents.director.DirectorAgent.resolve_style_conflict`
    to reconcile the scene's biblical / narrative intent with the
    high-energy kinetic directives encoded here.  The resulting conflict
    fragment is appended to ``scene.director_notes`` so that the
    ``CinematographerAgent`` and downstream ``ValidatorAgent`` can pick it up.

    Attributes:
        kinetic_level: Qualitative energy descriptor
            (e.g. ``"explosive/violent"``).
        shutter: Virtual shutter speed used to freeze motion
            (e.g. ``"1/1000"``).  A fast shutter eliminates motion blur and
            produces sharp, "freeze-frame" physics.
        physics_override: Mapping of physics-engine parameters to overriding
            values (e.g. ``{"fluid_turbulence": "chaotic high-pressure jets"}``).
            These are injected verbatim into the resolved prompt as texture
            descriptors.
    """

    kinetic_level: str
    shutter: str = "1/1000"
    physics_override: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kinetic_level:
            raise ValueError("StyleConflictMetadata.kinetic_level must not be empty.")


@dataclass(frozen=True)
class Character:
    """A character that appears in the production.

    Attributes:
        id: Unique stable identifier used to anchor visual consistency.
        name: Display name (may be used in prompts).
        description: Canonical visual description injected into shot prompts.
            If omitted, it is auto-built from *canonical_features*.
        role: Narrative role (e.g. "protagonist", "prophet", "narrator").
        canonical_features: Optional structured physical attributes
            (ethnicity, hair, eyes, build, distinguishing_marks, …).
        wardrobe_logic: Optional wardrobe state and per-environment
            interaction rules used by the consistency engine to inject
            context-sensitive rendering hints.
    """

    id: str
    name: str
    description: str = ""
    role: str = "supporting"
    canonical_features: dict[str, Any] = field(default_factory=dict)
    wardrobe_logic: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Character.id must not be empty.")
        if not self.description and self.canonical_features:
            object.__setattr__(self, "description", self._build_description())
        if not self.description:
            raise ValueError("Character.description must not be empty.")

    def _build_description(self) -> str:
        """Build a Veo-compatible description string from *canonical_features*."""
        cf = self.canonical_features
        parts: list[str] = []
        if ethnicity := cf.get("ethnicity"):
            parts.append(ethnicity.replace("_", " "))
        if hair := cf.get("hair"):
            parts.append(f"hair: {hair}")
        if eyes := cf.get("eyes"):
            parts.append(f"eyes: {eyes}")
        if build := cf.get("build"):
            parts.append(f"build: {build}")
        if marks := cf.get("distinguishing_marks"):
            parts.append(marks)
        if initial := self.wardrobe_logic.get("initial_state"):
            parts.append(f"wardrobe: {initial}")
        return ", ".join(filter(None, parts))


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
        consistency_anchors: Key/value pairs injected by the consistency engine.
        id: Auto-generated unique identifier.
    """

    scene_id: str
    index: int
    prompt: str
    negative_prompt: str = ""
    duration_seconds: float = 5.0
    camera_angle: CameraAngle = CameraAngle.WIDE
    consistency_anchors: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("Shot.prompt must not be empty.")


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
        genre: Optional genre definition whose visual DNA is injected into all shots.
        style_conflict: Optional kinetic override that forces the DirectorAgent
            to reconcile narrative intent with action-genre physics.  When set,
            the ValidatorAgent enforces a "Physically Legible Chaos" gate on
            every shot prompt.
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
    genre: Genre | None = None
    style_conflict: StyleConflictMetadata | None = None

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
