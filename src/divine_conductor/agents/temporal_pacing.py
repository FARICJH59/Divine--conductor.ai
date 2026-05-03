"""TemporalPacingAgent — modulates shot timing and injects motion flow anchors."""

from __future__ import annotations

import logging
from typing import Sequence

from divine_conductor.agents.base import BaseAgent
from divine_conductor.models.production import (
    ProductionState,
    Shot,
    StyleConflictMetadata,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flow anchor templates
# ---------------------------------------------------------------------------

# Directional motion hints injected into shot prompts to maintain vector
# consistency across block boundaries. The agent cycles through these based
# on the shot index so consecutive blocks share momentum cues.
_FLOW_ANCHORS: list[str] = [
    "maintain left-to-right momentum",
    "preserve upward vertical energy from previous shot",
    "sustain rightward lateral sweep, continuous horizontal flow",
    "carry forward depth-push movement from prior frame",
    "hold diagonal rise from lower-left to upper-right",
    "maintain slow outward zoom, expanding field of view",
    "continue clockwise rotation established in preceding shot",
    "preserve downward settling motion from previous frame",
]

# ---------------------------------------------------------------------------
# Motion-bucket adjustment per conflict type
# ---------------------------------------------------------------------------

# Conflicts that call for *reduced* motion (to smooth over jarring transitions)
_DAMPEN_CONFLICTS: frozenset[str] = frozenset(
    {"temporal", "tonal", "weather", "time_of_day"}
)

# Conflicts that call for *increased* motion (to energise static sequences)
_ENERGISE_CONFLICTS: frozenset[str] = frozenset({"motion_vector", "action_gap"})

_MOTION_DAMPEN_FACTOR: float = 0.75   # multiply motion_bucket to reduce motion
_MOTION_ENERGISE_FACTOR: float = 1.25  # multiply motion_bucket to boost motion

_DURATION_DAMPEN_FACTOR: float = 0.85  # shorten duration on severe conflict
_DURATION_EXTEND_FACTOR: float = 1.15  # extend duration on mild conflict

#: Default motion bucket used when a shot has not been explicitly configured.
DEFAULT_MOTION_BUCKET: int = 127

# Severity threshold above which a conflict is considered severe
_SEVERE_THRESHOLD: float = 0.6


class TemporalPacingAgent(BaseAgent):
    """Modulates shot pacing and injects directional flow anchors.

    The agent:
    1. Reads any :class:`~divine_conductor.models.production.StyleConflictMetadata`
       stored in ``state.metadata["style_conflicts"]`` and applies per-shot
       adjustments to ``duration_seconds`` and ``motion_bucket``.
    2. Injects a *flow anchor* string into every shot prompt to ensure
       vector consistency across production-block boundaries.

    Args:
        base_motion_bucket: Default motion intensity (1–255) applied when no
            conflict metadata is present.
    """

    name = "temporal_pacing_agent"

    def __init__(self, base_motion_bucket: int = 127) -> None:
        if not 1 <= base_motion_bucket <= 255:
            raise ValueError("base_motion_bucket must be between 1 and 255.")
        self._base_motion_bucket = base_motion_bucket

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def run(self, state: ProductionState) -> ProductionState:
        """Apply temporal pacing adjustments to all shots in *state*."""
        conflicts: list[StyleConflictMetadata] = state.metadata.get(
            "style_conflicts", []
        )
        # Build a lookup: shot index → conflict (use first conflict if global)
        conflict_map = self._build_conflict_map(conflicts, len(state.shots))

        adjusted: list[Shot] = []
        for shot in state.shots:
            conflict = conflict_map.get(shot.index)
            adjusted.append(self._adjust_shot(shot, conflict))

        state.shots = adjusted
        state.metadata["temporal_pacing_applied"] = True
        state.metadata["temporal_pacing_flow_anchors"] = len(adjusted)
        logger.info(
            "[%s] Applied pacing to %d shots with %d conflict(s).",
            self.name,
            len(adjusted),
            len(conflicts),
        )
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adjust_shot(
        self, shot: Shot, conflict: StyleConflictMetadata | None
    ) -> Shot:
        """Return a new ``Shot`` with modulated duration, motion_bucket, and
        an injected flow anchor."""
        duration = shot.duration_seconds
        bucket = shot.motion_bucket if shot.motion_bucket != DEFAULT_MOTION_BUCKET else self._base_motion_bucket

        if conflict is not None:
            duration, bucket = self._apply_conflict(duration, bucket, conflict)

        # Clamp bucket to valid range
        bucket = max(1, min(255, round(bucket)))

        # Inject flow anchor into the prompt
        anchor = self._select_flow_anchor(shot.index)
        new_prompt = f"{shot.prompt}, {anchor}"

        return Shot(
            scene_id=shot.scene_id,
            index=shot.index,
            prompt=new_prompt,
            negative_prompt=shot.negative_prompt,
            duration_seconds=round(duration, 2),
            camera_angle=shot.camera_angle,
            motion_bucket=bucket,
            consistency_anchors=dict(shot.consistency_anchors),
            id=shot.id,
        )

    def _apply_conflict(
        self,
        duration: float,
        bucket: int,
        conflict: StyleConflictMetadata,
    ) -> tuple[float, int]:
        """Return adjusted (duration, bucket) for the given conflict."""
        severity = conflict.severity

        if conflict.conflict_type in _DAMPEN_CONFLICTS:
            # Severe conflict → shorter, calmer shot to ease the transition
            if severity >= _SEVERE_THRESHOLD:
                duration *= _DURATION_DAMPEN_FACTOR
                bucket = int(bucket * _MOTION_DAMPEN_FACTOR)
            else:
                duration *= _DURATION_EXTEND_FACTOR  # give more time to settle
        elif conflict.conflict_type in _ENERGISE_CONFLICTS:
            bucket = int(bucket * _MOTION_ENERGISE_FACTOR)
        else:
            # Use the conflict's own suggested motion_bucket if provided
            if conflict.motion_bucket != DEFAULT_MOTION_BUCKET:
                bucket = conflict.motion_bucket

        return duration, bucket

    @staticmethod
    def _build_conflict_map(
        conflicts: Sequence[StyleConflictMetadata],
        n_shots: int,
    ) -> dict[int, StyleConflictMetadata]:
        """Map shot indices to the most relevant conflict.

        If exactly one conflict exists it is applied to every shot.
        If multiple conflicts are given they are distributed round-robin.
        """
        if not conflicts:
            return {}
        if len(conflicts) == 1:
            return {i: conflicts[0] for i in range(n_shots)}
        return {i: conflicts[i % len(conflicts)] for i in range(n_shots)}

    @staticmethod
    def _select_flow_anchor(shot_index: int) -> str:
        """Return the flow anchor string for this shot index."""
        return _FLOW_ANCHORS[shot_index % len(_FLOW_ANCHORS)]
