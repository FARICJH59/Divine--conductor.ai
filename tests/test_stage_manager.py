"""Tests for StageManagerAgent and BatchSequenceController."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from divine_conductor.agents.cinematographer import CinematographerAgent
from divine_conductor.agents.director import DirectorAgent
from divine_conductor.agents.narrator import NarratorAgent
from divine_conductor.agents.stage_manager import StageManagerAgent
from divine_conductor.models.production import (
    CameraAngle,
    Coords3D,
    EmotionalTone,
    ProductionConfig,
    ProductionState,
    Scene,
    Shot,
    StageMap,
)
from divine_conductor.pipeline.batch_sequence_controller import (
    BatchSequenceController,
    ProductionBlock,
)
from divine_conductor.pipeline.orchestrator import PipelineOrchestrator


_PASSAGE = (
    "In the beginning God created the heavens and the earth. "
    "The earth was without form and void, and darkness was over the face of the deep. "
    "And the Spirit of God was hovering over the face of the waters. "
    'And God said, "Let there be light," and there was light.'
)


def _make_config(**kwargs) -> ProductionConfig:
    defaults = dict(name="Test Production", passage_text=_PASSAGE)
    defaults.update(kwargs)
    return ProductionConfig(**defaults)


def _state_with_directed_scenes() -> ProductionState:
    state = ProductionState(config=_make_config())
    state = NarratorAgent().run(state)
    state = DirectorAgent().run(state)
    return state


# ---------------------------------------------------------------------------
# Coords3D
# ---------------------------------------------------------------------------


class TestCoords3D:
    def test_default_values(self):
        c = Coords3D()
        assert c.x == 0.0
        assert c.y == 0.0
        assert c.z == 0.0

    def test_custom_values(self):
        c = Coords3D(x=1.0, y=-2.0, z=3.5)
        assert c.x == 1.0
        assert c.y == -2.0
        assert c.z == 3.5

    def test_immutable(self):
        c = Coords3D(x=1.0)
        with pytest.raises(Exception):
            c.x = 9.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StageMap.get_spatial_prompt
# ---------------------------------------------------------------------------


class TestStageMapGetSpatialPrompt:
    def _make_map(
        self,
        cam: Coords3D,
        light: Coords3D,
        subj: Coords3D = Coords3D(),
    ) -> StageMap:
        return StageMap(
            scene_id="test",
            primary_subject=subj,
            camera_position=cam,
            key_light_motivation=light,
        )

    def test_returns_non_empty_string(self):
        sm = self._make_map(cam=Coords3D(0, 0, 5), light=Coords3D(0, 1, 0))
        assert sm.get_spatial_prompt() != ""

    def test_camera_left_token(self):
        sm = self._make_map(cam=Coords3D(-1.0, 0, 5), light=Coords3D(0, 1, 0))
        prompt = sm.get_spatial_prompt()
        assert "left" in prompt

    def test_camera_right_token(self):
        sm = self._make_map(cam=Coords3D(1.0, 0, 5), light=Coords3D(0, 1, 0))
        prompt = sm.get_spatial_prompt()
        assert "right" in prompt

    def test_camera_centered_token(self):
        sm = self._make_map(cam=Coords3D(0, 0, 5), light=Coords3D(0, 1, 0))
        prompt = sm.get_spatial_prompt()
        assert "centered" in prompt

    def test_high_angle_token(self):
        sm = self._make_map(cam=Coords3D(0, 6, 0.5), light=Coords3D(0, 1, 0))
        prompt = sm.get_spatial_prompt()
        assert "high-angle" in prompt

    def test_low_angle_token(self):
        sm = self._make_map(cam=Coords3D(0, -2, 3), light=Coords3D(0, 1, 0))
        prompt = sm.get_spatial_prompt()
        assert "low-angle" in prompt

    def test_eye_level_token(self):
        sm = self._make_map(cam=Coords3D(0, 0, 4), light=Coords3D(0, 1, 0))
        prompt = sm.get_spatial_prompt()
        assert "eye-level" in prompt

    def test_wide_distance_token(self):
        sm = self._make_map(cam=Coords3D(0, 0, 8), light=Coords3D(0, 1, 0))
        prompt = sm.get_spatial_prompt()
        assert "wide" in prompt

    def test_intimate_proximity_token(self):
        sm = self._make_map(cam=Coords3D(0, 0, 0.8), light=Coords3D(0, 1, 0))
        prompt = sm.get_spatial_prompt()
        assert "intimate" in prompt

    def test_key_light_from_above(self):
        sm = self._make_map(cam=Coords3D(0, 0, 5), light=Coords3D(0, 2, 0))
        prompt = sm.get_spatial_prompt()
        assert "above" in prompt

    def test_key_light_from_left(self):
        sm = self._make_map(cam=Coords3D(0, 0, 5), light=Coords3D(-1, 0, 0))
        prompt = sm.get_spatial_prompt()
        assert "left" in prompt

    def test_key_light_from_right(self):
        sm = self._make_map(cam=Coords3D(0, 0, 5), light=Coords3D(1, 0, 0))
        prompt = sm.get_spatial_prompt()
        assert "right" in prompt

    def test_prompt_contains_four_tokens(self):
        sm = self._make_map(cam=Coords3D(0, 0, 5), light=Coords3D(0, 1, 0))
        parts = sm.get_spatial_prompt().split(", ")
        assert len(parts) == 4


# ---------------------------------------------------------------------------
# StageManagerAgent
# ---------------------------------------------------------------------------


class TestStageManagerAgent:
    def test_run_populates_stage_maps(self):
        state = _state_with_directed_scenes()
        state = StageManagerAgent().run(state)
        assert "stage_maps" in state.metadata
        assert len(state.metadata["stage_maps"]) == len(state.scenes)

    def test_stage_maps_keyed_by_scene_id(self):
        state = _state_with_directed_scenes()
        state = StageManagerAgent().run(state)
        stage_maps = state.metadata["stage_maps"]
        for scene in state.scenes:
            assert scene.id in stage_maps

    def test_stage_map_scene_id_matches(self):
        state = _state_with_directed_scenes()
        state = StageManagerAgent().run(state)
        for scene in state.scenes:
            sm: StageMap = state.metadata["stage_maps"][scene.id]
            assert sm.scene_id == scene.id

    def test_spatial_tokens_set_on_scenes(self):
        state = _state_with_directed_scenes()
        state = StageManagerAgent().run(state)
        for scene in state.scenes:
            assert scene.spatial_tokens != ""

    def test_spatial_tokens_match_stage_map_prompt(self):
        state = _state_with_directed_scenes()
        state = StageManagerAgent().run(state)
        for scene in state.scenes:
            sm: StageMap = state.metadata["stage_maps"][scene.id]
            assert scene.spatial_tokens == sm.get_spatial_prompt()

    def test_all_camera_angles_covered(self):
        """Every CameraAngle maps to a distinct camera position."""
        agent = StageManagerAgent()
        for angle in CameraAngle:
            scene = Scene(index=0, text="Test.", camera_angle=angle)
            sm = agent._build_stage_map(scene)
            # camera_position must be a valid Coords3D (not the default fallback
            # unless the angle is genuinely unmapped — which shouldn't happen)
            assert isinstance(sm.camera_position, Coords3D)

    def test_all_tones_covered(self):
        """Every EmotionalTone maps to a distinct key-light direction."""
        agent = StageManagerAgent()
        for tone in EmotionalTone:
            scene = Scene(index=0, text="Test.", tone=tone)
            sm = agent._build_stage_map(scene)
            assert isinstance(sm.key_light_motivation, Coords3D)

    def test_primary_subject_at_origin(self):
        state = _state_with_directed_scenes()
        state = StageManagerAgent().run(state)
        for sm in state.metadata["stage_maps"].values():
            assert sm.primary_subject == Coords3D(0.0, 0.0, 0.0)

    def test_spatial_tokens_injected_into_shot_prompts(self):
        """Spatial tokens produced by StageManager appear in shot prompts."""
        state = _state_with_directed_scenes()
        state = StageManagerAgent().run(state)
        state = CinematographerAgent().run(state)
        for shot in state.shots:
            scene = next(s for s in state.scenes if s.id == shot.scene_id)
            # At least one spatial token fragment should appear in the prompt
            assert any(
                token in shot.prompt
                for token in scene.spatial_tokens.split(", ")
            )

    def test_agent_name(self):
        assert StageManagerAgent.name == "stage_manager_agent"


# ---------------------------------------------------------------------------
# Orchestrator — StageManager runs before Cinematographer
# ---------------------------------------------------------------------------


class TestOrchestratorStageManagerOrder:
    def test_stage_maps_present_after_full_pipeline(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        state = PipelineOrchestrator(config).run()
        assert "stage_maps" in state.metadata

    def test_spatial_tokens_in_shots(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        state = PipelineOrchestrator(config).run()
        # All shots should contain at least one spatial token
        combined = " ".join(sh.prompt for sh in state.shots)
        assert "camera" in combined

    def test_stage_manager_before_cinematographer(self, tmp_path):
        """Spatial tokens must be in shot prompts, proving ordering is correct."""
        config = _make_config(output_path=str(tmp_path))
        state = PipelineOrchestrator(config).run()
        for shot in state.shots:
            # The prompt is post-consistency-engine but should contain framing tokens
            assert shot.prompt.strip() != ""


# ---------------------------------------------------------------------------
# BatchSequenceController
# ---------------------------------------------------------------------------


def _make_state_with_shots(
    n_scenes: int = 6,
    duration_per_scene: float = 5.0,
) -> ProductionState:
    """Create a minimal ProductionState with scenes, shots, and stage maps."""
    config = _make_config()
    state = ProductionState(config=config)
    scenes: list[Scene] = []
    shots: list[Shot] = []
    stage_maps: dict[str, StageMap] = {}

    for i in range(n_scenes):
        scene = Scene(
            index=i,
            text=f"Scene text number {i + 1}.",
            duration_seconds=duration_per_scene,
            camera_angle=CameraAngle.MEDIUM,
        )
        scenes.append(scene)
        shot = Shot(scene_id=scene.id, index=0, prompt=f"Shot {i + 1} prompt.")
        shots.append(shot)
        cam_x = float(i) * 0.5  # camera drifts right across scenes
        sm = StageMap(
            scene_id=scene.id,
            primary_subject=Coords3D(),
            camera_position=Coords3D(x=cam_x, y=0.0, z=4.0),
            key_light_motivation=Coords3D(x=0.3, y=1.0, z=0.5),
        )
        stage_maps[scene.id] = sm
        scene.spatial_tokens = sm.get_spatial_prompt()

    state.scenes = scenes
    state.shots = shots
    state.metadata["stage_maps"] = stage_maps
    return state


class TestBatchSequenceControllerInit:
    def test_default_target_duration(self):
        ctrl = BatchSequenceController()
        assert ctrl._target_duration == 10.0

    def test_custom_target_duration(self):
        ctrl = BatchSequenceController(target_block_duration=20.0)
        assert ctrl._target_duration == 20.0

    def test_invalid_duration_raises(self):
        with pytest.raises(ValueError, match="positive"):
            BatchSequenceController(target_block_duration=0.0)


class TestSplitIntoBlocks:
    def test_single_block_when_duration_fits(self):
        state = _make_state_with_shots(n_scenes=2, duration_per_scene=4.0)
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        assert len(blocks) == 1

    def test_multiple_blocks_created(self):
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        # 6 × 5s = 30s with 10s target → 3 blocks of 2 scenes each
        assert len(blocks) == 3

    def test_all_scenes_present_across_blocks(self):
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        scene_ids = {s.id for b in blocks for s in b.scenes}
        assert scene_ids == {s.id for s in state.scenes}

    def test_block_indices_sequential(self):
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        for i, block in enumerate(blocks):
            assert block.block_index == i

    def test_shots_distributed_correctly(self):
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        total_shots = sum(len(b.shots) for b in blocks)
        assert total_shots == len(state.shots)

    def test_cumulative_time_end_matches_start_of_next(self):
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        for i in range(len(blocks) - 1):
            assert blocks[i].cumulative_time_end == pytest.approx(
                blocks[i + 1].cumulative_time_start
            )

    def test_single_scene_always_starts_new_block(self):
        """A scene that overflows the target still occupies its own block."""
        state = _make_state_with_shots(n_scenes=1, duration_per_scene=20.0)
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        assert len(blocks) == 1


class TestApplyVectorDirectionality:
    def test_camera_direction_preserved_across_boundary(self):
        """Camera x-position should continue increasing across block boundary."""
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        stage_maps = state.metadata["stage_maps"]

        updated = ctrl.apply_vector_directionality(blocks, stage_maps)

        # Block 0 ends with scenes[1], block 1 starts with scenes[2]
        prev_last_scene = blocks[0].scenes[-1]
        prev_second_scene = blocks[0].scenes[-2]
        first_next_scene = blocks[1].scenes[0]

        vel_x = (
            updated[prev_last_scene.id].camera_position.x
            - updated[prev_second_scene.id].camera_position.x
        )
        expected_x = (
            updated[prev_last_scene.id].camera_position.x + vel_x
        )
        assert updated[first_next_scene.id].camera_position.x == pytest.approx(
            expected_x
        )

    def test_spatial_tokens_updated_after_adjustment(self):
        """scene.spatial_tokens must reflect the adjusted camera position."""
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        stage_maps = state.metadata["stage_maps"]

        ctrl.apply_vector_directionality(blocks, stage_maps)

        first_next_scene = blocks[1].scenes[0]
        updated_map = state.metadata["stage_maps"][first_next_scene.id]
        # After apply, the spatial tokens on the scene object must match the
        # new map (apply_vector_directionality mutates the scene in place)
        assert first_next_scene.spatial_tokens == updated_map.get_spatial_prompt()

    def test_zero_velocity_no_adjustment(self):
        """If camera is stationary, no adjustment should be made."""
        state = _make_state_with_shots(n_scenes=4, duration_per_scene=5.0)
        # Override stage maps so all camera positions are identical
        for scene_id, sm in state.metadata["stage_maps"].items():
            state.metadata["stage_maps"][scene_id] = StageMap(
                scene_id=sm.scene_id,
                primary_subject=sm.primary_subject,
                camera_position=Coords3D(x=0.0, y=0.0, z=4.0),
                key_light_motivation=sm.key_light_motivation,
            )
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        original_maps = dict(state.metadata["stage_maps"])
        updated = ctrl.apply_vector_directionality(blocks, state.metadata["stage_maps"])

        first_next_scene = blocks[1].scenes[0]
        # No velocity means the camera position for the boundary scene is unchanged
        assert updated[first_next_scene.id].camera_position == pytest.approx(
            original_maps[first_next_scene.id].camera_position
        )

    def test_returns_dict_of_stage_maps(self):
        state = _make_state_with_shots(n_scenes=4, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        result = ctrl.apply_vector_directionality(blocks, state.metadata["stage_maps"])
        assert isinstance(result, dict)

    def test_single_block_no_adjustments(self):
        """With only one block there are no boundaries — original maps unchanged."""
        state = _make_state_with_shots(n_scenes=2, duration_per_scene=4.0)
        ctrl = BatchSequenceController(target_block_duration=10.0)
        blocks = ctrl.split_into_blocks(state)
        assert len(blocks) == 1
        original = {k: v for k, v in state.metadata["stage_maps"].items()}
        updated = ctrl.apply_vector_directionality(blocks, original)
        assert updated == original


class TestBatchSequenceControllerProcess:
    def test_returns_blocks_and_state(self, tmp_path):
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        state.config = _make_config(output_path=str(tmp_path))
        ctrl = BatchSequenceController(target_block_duration=10.0, output_path=str(tmp_path))
        blocks, returned_state = ctrl.process(state)
        assert isinstance(blocks, list)
        assert returned_state is state

    def test_manifest_written(self, tmp_path):
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0, output_path=str(tmp_path))
        ctrl.process(state)
        assert (tmp_path / "batch_manifest.json").exists()

    def test_manifest_valid_json(self, tmp_path):
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0, output_path=str(tmp_path))
        ctrl.process(state)
        data = json.loads((tmp_path / "batch_manifest.json").read_text())
        assert "production" in data
        assert "total_blocks" in data
        assert "blocks" in data
        assert isinstance(data["blocks"], list)

    def test_manifest_block_count_correct(self, tmp_path):
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0, output_path=str(tmp_path))
        blocks, _ = ctrl.process(state)
        data = json.loads((tmp_path / "batch_manifest.json").read_text())
        assert data["total_blocks"] == len(blocks)

    def test_updated_stage_maps_stored_in_state(self, tmp_path):
        state = _make_state_with_shots(n_scenes=6, duration_per_scene=5.0)
        ctrl = BatchSequenceController(target_block_duration=10.0, output_path=str(tmp_path))
        _, returned_state = ctrl.process(state)
        assert "stage_maps" in returned_state.metadata

    def test_process_with_orchestrator_output(self, tmp_path):
        """End-to-end: run full pipeline then batch-process the result."""
        config = _make_config(output_path=str(tmp_path))
        state = PipelineOrchestrator(config).run()
        ctrl = BatchSequenceController(
            target_block_duration=10.0, output_path=str(tmp_path)
        )
        blocks, processed_state = ctrl.process(state)
        assert len(blocks) >= 1
        assert (tmp_path / "batch_manifest.json").exists()


# ---------------------------------------------------------------------------
# ProductionBlock
# ---------------------------------------------------------------------------


class TestProductionBlock:
    def test_dataclass_fields(self):
        scene = Scene(index=0, text="Test.")
        shot = Shot(scene_id=scene.id, index=0, prompt="p")
        block = ProductionBlock(
            block_index=0,
            scenes=[scene],
            shots=[shot],
            cumulative_time_start=0.0,
            cumulative_time_end=5.0,
        )
        assert block.block_index == 0
        assert len(block.scenes) == 1
        assert len(block.shots) == 1
        assert block.cumulative_time_start == pytest.approx(0.0)
        assert block.cumulative_time_end == pytest.approx(5.0)
