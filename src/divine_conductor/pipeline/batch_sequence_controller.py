"""BatchSequenceController — processes a production as a sequence of blocks.

Long productions are split into discrete *ProductionBlocks* of approximately
``target_block_duration`` seconds each.  At every batch boundary the
controller reads the ``StageMap`` objects stored in
``state.metadata["stage_maps"]`` and enforces **Vector Directionality**:
if the camera was moving in a particular direction across the last two scenes
of block *N*, that same directional momentum is carried forward into the
first scene of block *N+1* by adjusting the stage map's camera position and
regenerating its spatial-token string.

After processing, a ``batch_manifest.json`` file is written to the
production's output directory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from divine_conductor.models.production import (
    Coords3D,
    ProductionState,
    Scene,
    Shot,
    StageMap,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProductionBlock:
    """A time-bounded slice of a production.

    Attributes:
        block_index: Zero-based index of this block in the sequence.
        scenes: Ordered scenes that belong to this block.
        shots: Shots that belong to this block.
        cumulative_time_start: Seconds of footage before this block starts.
        cumulative_time_end: Seconds of footage after this block ends.
    """

    block_index: int
    scenes: list[Scene]
    shots: list[Shot]
    cumulative_time_start: float
    cumulative_time_end: float


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class BatchSequenceController:
    """Splits a production into time-bounded blocks and enforces continuity.

    Args:
        target_block_duration: Preferred duration (seconds) for each block.
            Scenes are never split; a block extends beyond the target if the
            next scene would be the first one added.
        output_path: Directory where ``batch_manifest.json`` is written.

    Usage::

        controller = BatchSequenceController(
            target_block_duration=10.0,
            output_path="output",
        )
        blocks, state = controller.process(state)
    """

    def __init__(
        self,
        target_block_duration: float = 10.0,
        output_path: str = "output",
    ) -> None:
        if target_block_duration <= 0:
            raise ValueError("target_block_duration must be positive.")
        self._target_duration = target_block_duration
        self._output_path = Path(output_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_into_blocks(self, state: ProductionState) -> list[ProductionBlock]:
        """Partition *state*'s scenes into approximately equal-duration blocks.

        Returns a list of :class:`ProductionBlock` objects in scene order.
        """
        blocks: list[ProductionBlock] = []
        current_scenes: list[Scene] = []
        current_shots: list[Shot] = []
        current_duration = 0.0
        block_index = 0
        cumulative_start = 0.0

        for scene in state.scenes:
            scene_shots = [sh for sh in state.shots if sh.scene_id == scene.id]
            scene_duration = scene.duration_seconds

            # Start a new block if the current one would exceed the target,
            # but only if it already contains at least one scene.
            if current_scenes and (
                current_duration + scene_duration > self._target_duration
            ):
                blocks.append(
                    ProductionBlock(
                        block_index=block_index,
                        scenes=list(current_scenes),
                        shots=list(current_shots),
                        cumulative_time_start=cumulative_start,
                        cumulative_time_end=cumulative_start + current_duration,
                    )
                )
                cumulative_start += current_duration
                block_index += 1
                current_scenes = []
                current_shots = []
                current_duration = 0.0

            current_scenes.append(scene)
            current_shots.extend(scene_shots)
            current_duration += scene_duration

        if current_scenes:
            blocks.append(
                ProductionBlock(
                    block_index=block_index,
                    scenes=list(current_scenes),
                    shots=list(current_shots),
                    cumulative_time_start=cumulative_start,
                    cumulative_time_end=cumulative_start + current_duration,
                )
            )

        return blocks

    def apply_vector_directionality(
        self,
        blocks: list[ProductionBlock],
        stage_maps: dict[str, StageMap],
    ) -> dict[str, StageMap]:
        """Maintain camera Vector Directionality across batch boundaries.

        For each pair of consecutive blocks, the camera velocity is computed
        from the last two scenes of the preceding block.  That velocity is
        then applied to the first scene of the following block so that camera
        movement direction is preserved.

        The adjusted stage maps update ``scene.spatial_tokens`` in place so
        the effect is immediately visible in the shots that reference those
        scenes.

        Args:
            blocks: Ordered list of :class:`ProductionBlock` objects.
            stage_maps: Mapping of scene ID to :class:`StageMap` (as stored
                in ``state.metadata["stage_maps"]``).

        Returns:
            A new dict of stage maps with boundary adjustments applied.
        """
        updated_maps: dict[str, StageMap] = dict(stage_maps)

        for i in range(1, len(blocks)):
            prev_block = blocks[i - 1]
            curr_block = blocks[i]

            # Need at least two scenes in the previous block to compute velocity
            if len(prev_block.scenes) < 2:
                logger.debug(
                    "Block %d has fewer than 2 scenes; skipping vector carry-forward.",
                    prev_block.block_index,
                )
                continue

            last_scene = prev_block.scenes[-1]
            second_last_scene = prev_block.scenes[-2]

            map_last = updated_maps.get(last_scene.id)
            map_second_last = updated_maps.get(second_last_scene.id)

            if map_last is None or map_second_last is None:
                continue

            # Camera velocity vector (delta per scene-step)
            vel_x = map_last.camera_position.x - map_second_last.camera_position.x
            vel_y = map_last.camera_position.y - map_second_last.camera_position.y
            vel_z = map_last.camera_position.z - map_second_last.camera_position.z

            if vel_x == 0.0 and vel_y == 0.0 and vel_z == 0.0:
                logger.debug(
                    "Zero velocity at block %d → %d boundary; no adjustment needed.",
                    prev_block.block_index,
                    curr_block.block_index,
                )
                continue

            first_scene = curr_block.scenes[0]
            base_map = updated_maps.get(first_scene.id)
            if base_map is None:
                continue

            # Project new camera position continuing the velocity vector
            new_cam_pos = Coords3D(
                x=map_last.camera_position.x + vel_x,
                y=map_last.camera_position.y + vel_y,
                z=map_last.camera_position.z + vel_z,
            )
            adjusted_map = StageMap(
                scene_id=first_scene.id,
                primary_subject=base_map.primary_subject,
                camera_position=new_cam_pos,
                key_light_motivation=base_map.key_light_motivation,
            )
            updated_maps[first_scene.id] = adjusted_map
            # Refresh the scene's spatial tokens so shots reflect the change
            first_scene.spatial_tokens = adjusted_map.get_spatial_prompt()
            logger.debug(
                "Vector directionality applied at block boundary %d → %d: "
                "velocity=(%.2f, %.2f, %.2f)",
                prev_block.block_index,
                curr_block.block_index,
                vel_x,
                vel_y,
                vel_z,
            )

        return updated_maps

    def process(
        self, state: ProductionState
    ) -> tuple[list[ProductionBlock], ProductionState]:
        """Split *state* into blocks and apply cross-boundary continuity rules.

        Steps:
        1. Split scenes into :class:`ProductionBlock` objects.
        2. If stage maps are present in ``state.metadata``, apply Vector
           Directionality adjustment at every batch boundary.
        3. Write ``batch_manifest.json`` to the output directory.

        Args:
            state: A fully-processed ``ProductionState`` (post consistency engine).

        Returns:
            A tuple of ``(blocks, state)`` where *blocks* is the ordered list
            of :class:`ProductionBlock` objects and *state* is the (possibly
            mutated) production state with updated stage maps.
        """
        blocks = self.split_into_blocks(state)
        logger.info(
            "[batch_sequence_controller] Split '%s' into %d block(s).",
            state.config.name,
            len(blocks),
        )

        stage_maps: dict[str, StageMap] = state.metadata.get("stage_maps", {})
        if stage_maps and len(blocks) > 1:
            updated_maps = self.apply_vector_directionality(blocks, stage_maps)
            state.metadata["stage_maps"] = updated_maps

        self._write_manifest(blocks, state)
        return blocks, state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_manifest(
        self, blocks: list[ProductionBlock], state: ProductionState
    ) -> None:
        """Serialise block metadata to ``batch_manifest.json``."""
        self._output_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "production": state.config.name,
            "total_blocks": len(blocks),
            "target_block_duration": self._target_duration,
            "blocks": [
                {
                    "block_index": b.block_index,
                    "scene_count": len(b.scenes),
                    "shot_count": len(b.shots),
                    "cumulative_time_start": b.cumulative_time_start,
                    "cumulative_time_end": b.cumulative_time_end,
                }
                for b in blocks
            ],
        }
        out_path = self._output_path / "batch_manifest.json"
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("[batch_sequence_controller] Manifest written to %s", out_path)
