"""Tests for BatchSequenceController, TemporalPacingAgent, and ValidatorAgent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from divine_conductor.agents.temporal_pacing import TemporalPacingAgent, _FLOW_ANCHORS
from divine_conductor.agents.validator import (
    ValidatorAgent,
    ValidationIssue,
    HARD_RESET_INTERVAL_SECONDS,
)
from divine_conductor.models.production import (
    CameraAngle,
    Character,
    PalettePreset,
    ProductionConfig,
    ProductionState,
    Shot,
    StyleConflictMetadata,
)
from divine_conductor.pipeline.batch_sequence_controller import (
    BatchSequenceController,
    ContinuitySeed,
    ProductionBlock,
    BLOCK_DURATION_SECONDS,
)
from divine_conductor.pipeline.orchestrator import PipelineOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LONG_PASSAGE = (
    "In the beginning God created the heavens and the earth. "
    "The earth was without form and void, and darkness was over the face of the deep. "
    "And the Spirit of God was hovering over the face of the waters. "
    'And God said, "Let there be light," and there was light. '
    "And God saw that the light was good. "
    "And God separated the light from the darkness. "
    "God called the light Day, and the darkness he called Night. "
    "And there was evening and there was morning, the first day."
)

_SHORT_PASSAGE = "God created light. The waters were divided."


def _make_config(passage: str = _SHORT_PASSAGE, **kwargs) -> ProductionConfig:
    defaults = dict(name="Test Production", passage_text=passage)
    defaults.update(kwargs)
    return ProductionConfig(**defaults)


def _make_shot(
    index: int = 0,
    prompt: str = "Wide cinematic shot of the ancient desert landscape under golden light.",
    negative_prompt: str = "",
    duration_seconds: float = 5.0,
    motion_bucket: int = 127,
) -> Shot:
    return Shot(
        scene_id="scene-1",
        index=index,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration_seconds=duration_seconds,
        motion_bucket=motion_bucket,
    )


# ===========================================================================
# StyleConflictMetadata
# ===========================================================================


class TestStyleConflictMetadata:
    def test_valid_metadata(self):
        scm = StyleConflictMetadata(conflict_type="temporal", severity=0.5)
        assert scm.conflict_type == "temporal"
        assert scm.motion_bucket == 127

    def test_empty_conflict_type_raises(self):
        with pytest.raises(ValueError, match="conflict_type"):
            StyleConflictMetadata(conflict_type="")

    def test_severity_out_of_range_raises(self):
        with pytest.raises(ValueError, match="severity"):
            StyleConflictMetadata(conflict_type="tonal", severity=1.5)

    def test_motion_bucket_out_of_range_raises(self):
        with pytest.raises(ValueError, match="motion_bucket"):
            StyleConflictMetadata(conflict_type="tonal", motion_bucket=0)

    def test_motion_bucket_upper_bound(self):
        with pytest.raises(ValueError, match="motion_bucket"):
            StyleConflictMetadata(conflict_type="tonal", motion_bucket=256)

    def test_custom_motion_bucket(self):
        scm = StyleConflictMetadata(conflict_type="motion_vector", motion_bucket=200)
        assert scm.motion_bucket == 200


# ===========================================================================
# Shot.motion_bucket
# ===========================================================================


class TestShotMotionBucket:
    def test_default_motion_bucket(self):
        shot = _make_shot()
        assert shot.motion_bucket == 127

    def test_custom_motion_bucket(self):
        shot = _make_shot(motion_bucket=64)
        assert shot.motion_bucket == 64

    def test_invalid_motion_bucket_raises(self):
        with pytest.raises(ValueError, match="motion_bucket"):
            Shot(scene_id="s", index=0, prompt="p", motion_bucket=0)


# ===========================================================================
# TemporalPacingAgent
# ===========================================================================


class TestTemporalPacingAgent:
    def _state_with_shots(self, n: int = 3) -> ProductionState:
        config = _make_config()
        state = ProductionState(config=config)
        state.shots = [
            _make_shot(
                index=i,
                prompt=f"Wide cinematic shot of the ancient desert landscape under golden light, scene {i}.",
            )
            for i in range(n)
        ]
        return state

    def test_flow_anchors_injected(self):
        state = self._state_with_shots(3)
        state = TemporalPacingAgent().run(state)
        for shot in state.shots:
            assert any(anchor in shot.prompt for anchor in _FLOW_ANCHORS)

    def test_metadata_flag_set(self):
        state = self._state_with_shots(2)
        state = TemporalPacingAgent().run(state)
        assert state.metadata.get("temporal_pacing_applied") is True

    def test_shot_count_unchanged(self):
        state = self._state_with_shots(4)
        before = len(state.shots)
        state = TemporalPacingAgent().run(state)
        assert len(state.shots) == before

    def test_dampen_conflict_reduces_motion_bucket(self):
        state = self._state_with_shots(2)
        conflict = StyleConflictMetadata(
            conflict_type="temporal", severity=0.8, motion_bucket=127
        )
        state.metadata["style_conflicts"] = [conflict]
        agent = TemporalPacingAgent(base_motion_bucket=127)
        state = agent.run(state)
        for shot in state.shots:
            assert shot.motion_bucket < 127

    def test_energise_conflict_increases_motion_bucket(self):
        state = self._state_with_shots(2)
        conflict = StyleConflictMetadata(
            conflict_type="motion_vector", severity=0.5, motion_bucket=127
        )
        state.metadata["style_conflicts"] = [conflict]
        state = TemporalPacingAgent(base_motion_bucket=127).run(state)
        for shot in state.shots:
            assert shot.motion_bucket > 127

    def test_motion_bucket_clamped_to_valid_range(self):
        state = self._state_with_shots(2)
        conflict = StyleConflictMetadata(
            conflict_type="motion_vector", severity=0.9, motion_bucket=240
        )
        state.metadata["style_conflicts"] = [conflict]
        state = TemporalPacingAgent(base_motion_bucket=240).run(state)
        for shot in state.shots:
            assert 1 <= shot.motion_bucket <= 255

    def test_invalid_base_motion_bucket_raises(self):
        with pytest.raises(ValueError, match="base_motion_bucket"):
            TemporalPacingAgent(base_motion_bucket=0)

    def test_duration_modulated_by_dampen_conflict(self):
        state = self._state_with_shots(1)
        conflict = StyleConflictMetadata(
            conflict_type="temporal", severity=0.9
        )
        state.metadata["style_conflicts"] = [conflict]
        original_duration = state.shots[0].duration_seconds
        state = TemporalPacingAgent().run(state)
        assert state.shots[0].duration_seconds < original_duration

    def test_no_conflicts_does_not_crash(self):
        state = self._state_with_shots(3)
        state = TemporalPacingAgent().run(state)
        assert len(state.shots) == 3

    def test_flow_anchor_cycles_through_all(self):
        """Enough shots that every anchor in _FLOW_ANCHORS is used at least once."""
        n = len(_FLOW_ANCHORS)
        state = self._state_with_shots(n)
        state = TemporalPacingAgent().run(state)
        used_anchors = {
            anchor
            for shot in state.shots
            for anchor in _FLOW_ANCHORS
            if anchor in shot.prompt
        }
        assert len(used_anchors) == n


# ===========================================================================
# ValidatorAgent
# ===========================================================================


class TestValidatorAgent:
    def _make_state(self, shots: list[Shot]) -> ProductionState:
        state = ProductionState(config=_make_config())
        state.shots = shots
        return state

    def test_valid_shots_no_issues(self):
        shots = [_make_shot(index=i) for i in range(3)]
        state = self._make_state(shots)
        state = ValidatorAgent().run(state)
        issues = state.metadata.get("validation_issues", [])
        assert all(i["code"] != "short_prompt" for i in issues)

    def test_short_prompt_flagged(self):
        shots = [Shot(scene_id="s", index=0, prompt="short")]
        state = self._make_state(shots)
        state = ValidatorAgent().run(state)
        codes = [i["code"] for i in state.metadata["validation_issues"]]
        assert "short_prompt" in codes

    def test_hard_reset_check_low_diversity(self):
        shots = [Shot(scene_id="s", index=0, prompt="the the the")]
        issues = ValidatorAgent().hard_reset_check(shots)
        codes = [i.code for i in issues]
        assert "low_lexical_diversity" in codes

    def test_hard_reset_check_missing_negative_prompt(self):
        shots = [_make_shot(negative_prompt="")]
        issues = ValidatorAgent().hard_reset_check(shots)
        codes = [i.code for i in issues]
        assert "missing_negative_prompt" in codes

    def test_hard_reset_check_good_shot_no_negative_issue_only(self):
        shots = [
            Shot(
                scene_id="s",
                index=0,
                prompt="Wide cinematic shot of the ancient desert landscape under golden light.",
                negative_prompt="blurry, low quality",
            )
        ]
        issues = ValidatorAgent().hard_reset_check(shots)
        assert not any(i.code == "low_lexical_diversity" for i in issues)

    def test_hard_reset_updates_state_metadata(self):
        state = ProductionState(config=_make_config())
        shots = [Shot(scene_id="s", index=0, prompt="a a a")]
        ValidatorAgent().hard_reset_check(shots, state)
        assert "validation_issues" in state.metadata

    def test_hard_reset_issue_flag(self):
        shots = [Shot(scene_id="s", index=0, prompt="a b c")]
        issues = ValidatorAgent().hard_reset_check(shots)
        assert all(i.hard_reset for i in issues)

    def test_validation_issue_to_dict(self):
        issue = ValidationIssue(shot_id="abc", code="test", detail="desc")
        d = issue.to_dict()
        assert d["shot_id"] == "abc"
        assert d["code"] == "test"
        assert d["hard_reset"] is False


# ===========================================================================
# BatchSequenceController
# ===========================================================================


class TestBatchSequenceController:
    def test_returns_list_of_blocks(self, tmp_path):
        config = _make_config(passage=_SHORT_PASSAGE, output_path=str(tmp_path))
        controller = BatchSequenceController(config)
        blocks = controller.run()
        assert isinstance(blocks, list)
        assert len(blocks) >= 1

    def test_each_block_has_shots(self, tmp_path):
        config = _make_config(passage=_SHORT_PASSAGE, output_path=str(tmp_path))
        blocks = BatchSequenceController(config).run()
        for block in blocks:
            assert len(block.shots) > 0

    def test_blocks_indexed_sequentially(self, tmp_path):
        config = _make_config(passage=_LONG_PASSAGE, output_path=str(tmp_path))
        blocks = BatchSequenceController(config).run()
        for i, block in enumerate(blocks):
            assert block.index == i

    def test_all_shots_flattens_blocks(self, tmp_path):
        config = _make_config(passage=_LONG_PASSAGE, output_path=str(tmp_path))
        controller = BatchSequenceController(config)
        blocks = controller.run()
        all_shots = controller.all_shots(blocks)
        expected = sum(len(b.shots) for b in blocks)
        assert len(all_shots) == expected

    def test_manifest_written(self, tmp_path):
        config = _make_config(passage=_SHORT_PASSAGE, output_path=str(tmp_path))
        BatchSequenceController(config).run()
        manifest_path = tmp_path / "batch_manifest.json"
        assert manifest_path.exists()

    def test_manifest_valid_json(self, tmp_path):
        config = _make_config(passage=_SHORT_PASSAGE, output_path=str(tmp_path))
        BatchSequenceController(config).run()
        data = json.loads((tmp_path / "batch_manifest.json").read_text())
        assert "blocks" in data
        assert "total_duration_seconds" in data
        assert data["block_count"] == len(data["blocks"])

    def test_manifest_entry_has_shot_ids(self, tmp_path):
        config = _make_config(passage=_SHORT_PASSAGE, output_path=str(tmp_path))
        blocks = BatchSequenceController(config).run()
        for block in blocks:
            assert "shot_ids" in block.manifest_entry
            assert len(block.manifest_entry["shot_ids"]) == len(block.shots)

    def test_recursive_continuity_seeds_first_shot(self, tmp_path):
        """Block N+1's first shot prompt should reference the previous block."""
        config = _make_config(passage=_LONG_PASSAGE, output_path=str(tmp_path))
        blocks = BatchSequenceController(config).run()
        if len(blocks) < 2:
            pytest.skip("Need at least 2 blocks for this test.")
        second_block_first_shot = blocks[1].shots[0]
        assert "Continuing from block" in second_block_first_shot.prompt

    def test_continuity_seed_extracted(self, tmp_path):
        config = _make_config(passage=_SHORT_PASSAGE, output_path=str(tmp_path))
        blocks = BatchSequenceController(config).run()
        # Every block except possibly the last should have a continuity seed
        for block in blocks:
            assert block.continuity_seed is not None or block.index == blocks[-1].index

    def test_long_passage_multiple_blocks(self, tmp_path):
        config = _make_config(passage=_LONG_PASSAGE, output_path=str(tmp_path))
        blocks = BatchSequenceController(config).run()
        assert len(blocks) >= 2

    def test_custom_block_duration(self, tmp_path):
        config = _make_config(passage=_LONG_PASSAGE, output_path=str(tmp_path))
        # Very small block duration → more blocks
        blocks_small = BatchSequenceController(
            config, block_duration=5.0
        ).run()
        blocks_large = BatchSequenceController(
            config, block_duration=20.0
        ).run()
        assert len(blocks_small) >= len(blocks_large)

    def test_blocks_cover_all_passage_text(self, tmp_path):
        """Every sentence from the passage appears in at least one block excerpt."""
        config = _make_config(passage=_SHORT_PASSAGE, output_path=str(tmp_path))
        blocks = BatchSequenceController(config).run()
        combined = " ".join(b.passage_excerpt for b in blocks)
        # Check a distinctive word from each sentence
        assert "created" in combined
        assert "divided" in combined


# ===========================================================================
# PipelineOrchestrator.run_batch
# ===========================================================================


class TestOrchestratorRunBatch:
    def test_run_batch_returns_blocks_and_state(self, tmp_path):
        config = _make_config(passage=_LONG_PASSAGE, output_path=str(tmp_path))
        orch = PipelineOrchestrator(config)
        blocks, state = orch.run_batch()
        assert isinstance(blocks, list)
        assert len(state.shots) > 0

    def test_run_batch_writes_output(self, tmp_path):
        config = _make_config(passage=_LONG_PASSAGE, output_path=str(tmp_path))
        PipelineOrchestrator(config).run_batch()
        assert (tmp_path / "shots.json").exists()
        assert (tmp_path / "batch_manifest.json").exists()

    def test_run_batch_json_contains_motion_bucket(self, tmp_path):
        config = _make_config(passage=_SHORT_PASSAGE, output_path=str(tmp_path))
        PipelineOrchestrator(config).run_batch()
        data = json.loads((tmp_path / "shots.json").read_text())
        for shot in data["shots"]:
            assert "motion_bucket" in shot

    def test_run_still_works_after_changes(self, tmp_path):
        """Ensure the original single-block run() still passes end-to-end."""
        config = _make_config(output_path=str(tmp_path))
        state = PipelineOrchestrator(config).run()
        assert len(state.shots) > 0

    def test_run_json_contains_motion_bucket(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        PipelineOrchestrator(config).run()
        data = json.loads((tmp_path / "shots.json").read_text())
        for shot in data["shots"]:
            assert "motion_bucket" in shot
            assert 1 <= shot["motion_bucket"] <= 255
