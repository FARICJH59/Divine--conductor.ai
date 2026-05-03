"""Tests for AuralAgent, AudioPlan, LoudnessGate, and orchestrator integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from divine_conductor.agents.aural import (
    AMBIENT_TARGET_LUFS,
    DIEGETIC_TARGET_LUFS,
    NARRATOR_LUFS,
    AudioPlan,
    AuralAgent,
    KineticLevel,
    LoudnessGate,
    _build_diegetic_hits,
)
from divine_conductor.agents.cinematographer import CinematographerAgent
from divine_conductor.agents.director import DirectorAgent
from divine_conductor.agents.narrator import NarratorAgent
from divine_conductor.models.production import (
    CameraAngle,
    EmotionalTone,
    ProductionConfig,
    ProductionState,
    Scene,
    Shot,
)
from divine_conductor.pipeline.orchestrator import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASSAGE = (
    "In the beginning God created the heavens and the earth. "
    "And God said, let there be light, and there was light. "
    "The enemy trembled with fear and battled against them."
)


def _make_config(**kwargs) -> ProductionConfig:
    defaults = dict(name="Aural Test", passage_text=_PASSAGE)
    defaults.update(kwargs)
    return ProductionConfig(**defaults)


def _make_pipeline_state() -> ProductionState:
    """Run Narrator → Director → Cinematographer to produce a full shot list."""
    state = ProductionState(config=_make_config())
    state = NarratorAgent().run(state)
    state = DirectorAgent().run(state)
    state = CinematographerAgent().run(state)
    return state


def _make_shot(
    camera_angle: CameraAngle = CameraAngle.WIDE,
    duration: float = 5.0,
) -> Shot:
    return Shot(
        scene_id="test-scene",
        index=0,
        prompt="test prompt",
        camera_angle=camera_angle,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# AudioPlan
# ---------------------------------------------------------------------------


class TestAudioPlan:
    def test_to_dict_has_required_keys(self):
        plan = AudioPlan(
            shot_id="abc",
            ambient_layer="sacred_choir_drone",
            diegetic_hits=[],
            musical_prompt="test prompt",
            kinetic_level=KineticLevel.LOW,
            duration_seconds=5.0,
        )
        d = plan.to_dict()
        for key in (
            "shot_id",
            "ambient_layer",
            "diegetic_hits",
            "musical_prompt",
            "kinetic_level",
            "duration_seconds",
        ):
            assert key in d

    def test_to_dict_kinetic_level_is_string(self):
        plan = AudioPlan(
            shot_id="x",
            ambient_layer="pad",
            kinetic_level=KineticLevel.HIGH,
        )
        assert plan.to_dict()["kinetic_level"] == "high"

    def test_default_diegetic_hits_is_empty_list(self):
        plan = AudioPlan(shot_id="y", ambient_layer="pad")
        assert plan.diegetic_hits == []


# ---------------------------------------------------------------------------
# _build_diegetic_hits
# ---------------------------------------------------------------------------


class TestBuildDiegeticHits:
    def test_low_kinetic_returns_empty(self):
        hits = _build_diegetic_hits(KineticLevel.LOW, 5.0)
        assert hits == []

    def test_medium_kinetic_returns_one_hit(self):
        hits = _build_diegetic_hits(KineticLevel.MEDIUM, 5.0)
        assert len(hits) == 1
        assert hits[0]["effect"] == "ambient_transition"

    def test_high_kinetic_returns_two_hits(self):
        hits = _build_diegetic_hits(KineticLevel.HIGH, 10.0)
        assert len(hits) == 2

    def test_high_kinetic_hit_timing(self):
        hits = _build_diegetic_hits(KineticLevel.HIGH, 8.0)
        assert hits[0]["time_seconds"] == 0.0
        assert hits[1]["time_seconds"] == pytest.approx(4.0)

    def test_high_kinetic_first_hit_is_impact(self):
        hits = _build_diegetic_hits(KineticLevel.HIGH, 5.0)
        assert hits[0]["effect"] == "impact_hit"

    def test_high_kinetic_second_hit_is_riser(self):
        hits = _build_diegetic_hits(KineticLevel.HIGH, 5.0)
        assert hits[1]["effect"] == "tension_riser"

    def test_medium_hit_has_gain_db(self):
        hits = _build_diegetic_hits(KineticLevel.MEDIUM, 5.0)
        assert "gain_db" in hits[0]


# ---------------------------------------------------------------------------
# LoudnessGate
# ---------------------------------------------------------------------------


class TestLoudnessGate:
    def test_default_narrator_lufs(self):
        gate = LoudnessGate()
        assert gate.narrator_lufs == NARRATOR_LUFS

    def test_narrator_above_ambient(self):
        gate = LoudnessGate()
        assert gate.narrator_lufs > gate.ambient_lufs

    def test_narrator_above_diegetic(self):
        gate = LoudnessGate()
        assert gate.narrator_lufs > gate.diegetic_lufs

    def test_headroom_db_positive(self):
        gate = LoudnessGate()
        assert gate.narrator_headroom_db() > 0

    def test_headroom_db_value(self):
        gate = LoudnessGate()
        expected = NARRATOR_LUFS - AMBIENT_TARGET_LUFS
        assert gate.narrator_headroom_db() == pytest.approx(expected)

    def test_gate_audio_plan_keys(self):
        gate = LoudnessGate()
        plan = AudioPlan(shot_id="s1", ambient_layer="pad")
        result = gate.gate_audio_plan(plan)
        for key in (
            "shot_id",
            "narrator_lufs",
            "ambient_layer_gain_db",
            "diegetic_gain_db",
            "mix_note",
        ):
            assert key in result

    def test_gate_ambient_gain_is_negative(self):
        gate = LoudnessGate()
        plan = AudioPlan(shot_id="s1", ambient_layer="pad")
        result = gate.gate_audio_plan(plan)
        assert result["ambient_layer_gain_db"] < 0

    def test_gate_diegetic_gain_is_negative(self):
        gate = LoudnessGate()
        plan = AudioPlan(shot_id="s1", ambient_layer="pad")
        result = gate.gate_audio_plan(plan)
        assert result["diegetic_gain_db"] < 0

    def test_gate_mix_note_mentions_narrator_lufs(self):
        gate = LoudnessGate()
        plan = AudioPlan(shot_id="s1", ambient_layer="pad")
        note = gate.gate_audio_plan(plan)["mix_note"]
        assert "-14" in note

    def test_invalid_gate_ambient_louder_than_narrator(self):
        with pytest.raises(ValueError, match="narrator_lufs"):
            LoudnessGate(narrator_lufs=-24.0, ambient_lufs=-14.0)

    def test_invalid_gate_diegetic_louder_than_narrator(self):
        with pytest.raises(ValueError, match="narrator_lufs"):
            LoudnessGate(narrator_lufs=-20.0, diegetic_lufs=-14.0)

    def test_custom_gate_values(self):
        gate = LoudnessGate(narrator_lufs=-16.0, ambient_lufs=-30.0, diegetic_lufs=-24.0)
        assert gate.narrator_lufs == -16.0
        assert gate.narrator_headroom_db() == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# AuralAgent
# ---------------------------------------------------------------------------


class TestAuralAgent:
    def test_produces_audio_plan_per_shot(self):
        state = _make_pipeline_state()
        n_shots = len(state.shots)
        state = AuralAgent().run(state)
        assert len(state.metadata["audio_plans"]) == n_shots

    def test_audio_plans_stored_in_metadata(self):
        state = _make_pipeline_state()
        state = AuralAgent().run(state)
        assert "audio_plans" in state.metadata
        assert isinstance(state.metadata["audio_plans"], list)

    def test_loudness_directives_stored_in_metadata(self):
        state = _make_pipeline_state()
        state = AuralAgent().run(state)
        assert "loudness_directives" in state.metadata
        assert len(state.metadata["loudness_directives"]) == len(state.shots)

    def test_narrator_lufs_stored_in_metadata(self):
        state = _make_pipeline_state()
        state = AuralAgent().run(state)
        assert state.metadata["aural_narrator_lufs"] == pytest.approx(NARRATOR_LUFS)

    def test_each_plan_has_shot_id(self):
        state = _make_pipeline_state()
        shot_ids = {sh.id for sh in state.shots}
        state = AuralAgent().run(state)
        plan_ids = {p["shot_id"] for p in state.metadata["audio_plans"]}
        assert plan_ids == shot_ids

    def test_each_plan_has_non_empty_ambient_layer(self):
        state = _make_pipeline_state()
        state = AuralAgent().run(state)
        for plan in state.metadata["audio_plans"]:
            assert plan["ambient_layer"] != ""

    def test_each_plan_has_non_empty_musical_prompt(self):
        state = _make_pipeline_state()
        state = AuralAgent().run(state)
        for plan in state.metadata["audio_plans"]:
            assert plan["musical_prompt"] != ""

    def test_dutch_angle_shot_has_high_kinetic(self):
        state = ProductionState(config=_make_config())
        state.scenes = [Scene(index=0, text="A tense battle scene.")]
        state.shots = [
            Shot(
                scene_id=state.scenes[0].id,
                index=0,
                prompt="Dutch angle, tension",
                camera_angle=CameraAngle.DUTCH_ANGLE,
                duration_seconds=5.0,
            )
        ]
        state = AuralAgent().run(state)
        plan = state.metadata["audio_plans"][0]
        assert plan["kinetic_level"] == "high"

    def test_high_kinetic_shot_has_diegetic_hits(self):
        state = ProductionState(config=_make_config())
        state.scenes = [Scene(index=0, text="Chaotic action.")]
        state.shots = [
            Shot(
                scene_id=state.scenes[0].id,
                index=0,
                prompt="Dutch angle action",
                camera_angle=CameraAngle.DUTCH_ANGLE,
                duration_seconds=6.0,
            )
        ]
        state = AuralAgent().run(state)
        plan = state.metadata["audio_plans"][0]
        assert len(plan["diegetic_hits"]) == 2

    def test_wide_angle_shot_has_no_diegetic_hits(self):
        state = ProductionState(config=_make_config())
        state.scenes = [Scene(index=0, text="Peaceful landscape.")]
        state.shots = [
            Shot(
                scene_id=state.scenes[0].id,
                index=0,
                prompt="Wide peaceful landscape",
                camera_angle=CameraAngle.WIDE,
                duration_seconds=5.0,
            )
        ]
        state = AuralAgent().run(state)
        plan = state.metadata["audio_plans"][0]
        assert plan["diegetic_hits"] == []

    def test_low_kinetic_angle_has_no_diegetic_hits(self):
        state = ProductionState(config=_make_config())
        state.scenes = [Scene(index=0, text="Overhead drone view.")]
        state.shots = [
            Shot(
                scene_id=state.scenes[0].id,
                index=0,
                prompt="Overhead drone view",
                camera_angle=CameraAngle.OVERHEAD,
                duration_seconds=5.0,
            )
        ]
        state = AuralAgent().run(state)
        plan = state.metadata["audio_plans"][0]
        assert plan["kinetic_level"] == "low"
        assert plan["diegetic_hits"] == []

    def test_tense_tone_gets_tension_ambient(self):
        state = ProductionState(config=_make_config())
        scene = Scene(index=0, text="The enemy trembled with fear.", tone=EmotionalTone.TENSE)
        state.scenes = [scene]
        state.shots = [
            Shot(
                scene_id=scene.id,
                index=0,
                prompt="Tense battle",
                camera_angle=CameraAngle.DUTCH_ANGLE,
                duration_seconds=5.0,
            )
        ]
        state = AuralAgent().run(state)
        plan = state.metadata["audio_plans"][0]
        assert plan["ambient_layer"] == "low_frequency_tension_pad"

    def test_reverent_tone_gets_sacred_choir_ambient(self):
        state = ProductionState(config=_make_config())
        scene = Scene(index=0, text="In holy reverence.", tone=EmotionalTone.REVERENT)
        state.scenes = [scene]
        state.shots = [
            Shot(
                scene_id=scene.id,
                index=0,
                prompt="Sacred reverence",
                camera_angle=CameraAngle.WIDE,
                duration_seconds=5.0,
            )
        ]
        state = AuralAgent().run(state)
        plan = state.metadata["audio_plans"][0]
        assert plan["ambient_layer"] == "sacred_choir_drone"

    def test_musical_prompt_matches_tense_tone(self):
        state = ProductionState(config=_make_config())
        scene = Scene(index=0, text="Enemy attack.", tone=EmotionalTone.TENSE)
        state.scenes = [scene]
        state.shots = [
            Shot(
                scene_id=scene.id,
                index=0,
                prompt="Attack scene",
                camera_angle=CameraAngle.DUTCH_ANGLE,
                duration_seconds=5.0,
            )
        ]
        state = AuralAgent().run(state)
        plan = state.metadata["audio_plans"][0]
        assert "suspenseful" in plan["musical_prompt"].lower()

    def test_custom_loudness_gate_reflected_in_metadata(self):
        gate = LoudnessGate(narrator_lufs=-16.0, ambient_lufs=-28.0, diegetic_lufs=-22.0)
        state = _make_pipeline_state()
        state = AuralAgent(loudness_gate=gate).run(state)
        assert state.metadata["aural_narrator_lufs"] == pytest.approx(-16.0)

    def test_loudness_directives_ambient_gain_negative(self):
        state = _make_pipeline_state()
        state = AuralAgent().run(state)
        for directive in state.metadata["loudness_directives"]:
            assert directive["ambient_layer_gain_db"] < 0

    def test_empty_shots_produces_empty_plans(self):
        state = ProductionState(config=_make_config())
        state.shots = []
        state = AuralAgent().run(state)
        assert state.metadata["audio_plans"] == []
        assert state.metadata["loudness_directives"] == []

    def test_shots_without_matching_scene_use_reverent_default(self):
        state = ProductionState(config=_make_config())
        state.scenes = []
        state.shots = [
            Shot(scene_id="nonexistent", index=0, prompt="Orphan shot")
        ]
        state = AuralAgent().run(state)
        plan = state.metadata["audio_plans"][0]
        # Reverent is the fallback; its ambient should be sacred_choir_drone
        assert plan["ambient_layer"] == "sacred_choir_drone"


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


class TestOrchestratorAuralIntegration:
    def test_batch_manifest_written(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        PipelineOrchestrator(config).run()
        assert (tmp_path / "batch_manifest.json").exists()

    def test_batch_manifest_is_valid_json(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        PipelineOrchestrator(config).run()
        data = json.loads((tmp_path / "batch_manifest.json").read_text())
        assert isinstance(data, dict)

    def test_batch_manifest_has_entries(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        state = PipelineOrchestrator(config).run()
        data = json.loads((tmp_path / "batch_manifest.json").read_text())
        assert len(data["entries"]) == len(state.shots)

    def test_batch_manifest_entries_have_audio_id(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        PipelineOrchestrator(config).run()
        data = json.loads((tmp_path / "batch_manifest.json").read_text())
        for entry in data["entries"]:
            assert "audio_id" in entry
            assert entry["audio_id"].startswith("audio_")

    def test_batch_manifest_has_narrator_lufs(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        PipelineOrchestrator(config).run()
        data = json.loads((tmp_path / "batch_manifest.json").read_text())
        assert data["narrator_lufs"] == pytest.approx(NARRATOR_LUFS)

    def test_batch_manifest_entries_have_loudness(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        PipelineOrchestrator(config).run()
        data = json.loads((tmp_path / "batch_manifest.json").read_text())
        for entry in data["entries"]:
            assert "loudness" in entry
            assert "narrator_lufs" in entry["loudness"]

    def test_batch_manifest_production_name(self, tmp_path):
        config = _make_config(output_path=str(tmp_path), name="Genesis Test")
        PipelineOrchestrator(config).run()
        data = json.loads((tmp_path / "batch_manifest.json").read_text())
        assert data["production"] == "Genesis Test"

    def test_aural_plans_present_in_state_metadata(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        state = PipelineOrchestrator(config).run()
        assert "audio_plans" in state.metadata
        assert len(state.metadata["audio_plans"]) > 0

    def test_pipeline_shot_count_unchanged_by_aural(self, tmp_path):
        """AuralAgent must not add or remove shots."""
        config = _make_config(output_path=str(tmp_path))
        state = PipelineOrchestrator(config).run()
        assert len(state.shots) > 0
        assert len(state.metadata["audio_plans"]) == len(state.shots)
