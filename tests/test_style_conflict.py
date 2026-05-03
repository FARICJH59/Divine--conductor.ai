"""Tests for the Style Conflict Resolver system.

Covers:
- StyleConflictMetadata model
- DirectorAgent.resolve_style_conflict static method
- DirectorAgent kinetic injection into director_notes
- ValidatorAgent blur / particle / consistency checks
- PipelineOrchestrator end-to-end with style_conflict active
- YAML parsing of style_conflict section
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from divine_conductor.agents.director import DirectorAgent
from divine_conductor.agents.narrator import NarratorAgent
from divine_conductor.agents.validator import ValidatorAgent
from divine_conductor.consistency.veo_consistency import Veo3ConsistencyEngine
from divine_conductor.models.production import (
    CameraAngle,
    EmotionalTone,
    PalettePreset,
    ProductionConfig,
    ProductionState,
    Scene,
    Shot,
    StyleConflictMetadata,
)
from divine_conductor.pipeline.orchestrator import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GENESIS_PASSAGE = (
    "In the beginning God created the heavens and the earth. "
    "Continental plates ruptured and mountains shattered into the sky. "
    "The waters were separated with violent force."
)

_ACTION_GENESIS_PASSAGE = (
    "The earth fractured. Debris filled the sky. "
    "Fragmentation of stone and shards of ice rained down. "
    "Nothing was serene — only explosive chaos."
)


def _make_config(**kwargs) -> ProductionConfig:
    defaults = dict(name="Conflict Test", passage_text=_GENESIS_PASSAGE)
    defaults.update(kwargs)
    return ProductionConfig(**defaults)


def _conflict() -> StyleConflictMetadata:
    return StyleConflictMetadata(
        kinetic_level="explosive/violent",
        shutter="1/1000",
        physics_override={"fluid_turbulence": "chaotic high-pressure jets"},
    )


def _make_scene(text: str = "The mountains shattered upward.") -> Scene:
    return Scene(index=0, text=text, tone=EmotionalTone.AWESTRUCK)


def _make_shot(prompt: str, scene_id: str = "s1") -> Shot:
    return Shot(scene_id=scene_id, index=0, prompt=prompt)


# ---------------------------------------------------------------------------
# StyleConflictMetadata model
# ---------------------------------------------------------------------------


class TestStyleConflictMetadata:
    def test_valid_metadata(self):
        sc = _conflict()
        assert sc.kinetic_level == "explosive/violent"
        assert sc.shutter == "1/1000"
        assert sc.physics_override["fluid_turbulence"] == "chaotic high-pressure jets"

    def test_empty_kinetic_level_raises(self):
        with pytest.raises(ValueError, match="kinetic_level"):
            StyleConflictMetadata(kinetic_level="")

    def test_defaults(self):
        sc = StyleConflictMetadata(kinetic_level="moderate")
        assert sc.shutter == "1/1000"
        assert sc.physics_override == {}

    def test_custom_shutter(self):
        sc = StyleConflictMetadata(kinetic_level="high", shutter="1/2000")
        assert sc.shutter == "1/2000"

    def test_frozen(self):
        sc = StyleConflictMetadata(kinetic_level="high")
        with pytest.raises((AttributeError, TypeError)):
            sc.kinetic_level = "low"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DirectorAgent.resolve_style_conflict
# ---------------------------------------------------------------------------


class TestResolveStyleConflict:
    def test_returns_dict_with_required_keys(self):
        scene = _make_scene()
        directive = DirectorAgent.resolve_style_conflict(scene, _conflict())
        assert set(directive.keys()) == {"subject", "energy", "fidelity", "texture"}

    def test_subject_derived_from_scene_text(self):
        scene = _make_scene("Continental plates rupturing violently.")
        directive = DirectorAgent.resolve_style_conflict(scene, _conflict())
        assert "Continental plates" in directive["subject"]

    def test_energy_matches_kinetic_level(self):
        directive = DirectorAgent.resolve_style_conflict(_make_scene(), _conflict())
        assert directive["energy"] == "explosive/violent"

    def test_fidelity_contains_shutter(self):
        directive = DirectorAgent.resolve_style_conflict(_make_scene(), _conflict())
        assert "1/1000" in directive["fidelity"]
        assert "no motion blur" in directive["fidelity"]

    def test_texture_uses_physics_override(self):
        directive = DirectorAgent.resolve_style_conflict(_make_scene(), _conflict())
        assert "chaotic high-pressure jets" in directive["texture"]

    def test_texture_fallback_without_physics_override(self):
        sc = StyleConflictMetadata(kinetic_level="violent")
        directive = DirectorAgent.resolve_style_conflict(_make_scene(), sc)
        assert "fragmentation" in directive["texture"]

    def test_subject_truncated_at_80_chars(self):
        long_text = "A" * 200
        scene = _make_scene(long_text)
        directive = DirectorAgent.resolve_style_conflict(scene, _conflict())
        assert len(directive["subject"]) <= 80


# ---------------------------------------------------------------------------
# DirectorAgent — conflict injection into director_notes
# ---------------------------------------------------------------------------


class TestDirectorConflictInjection:
    def _run_director(self, style_conflict=None, passage=_GENESIS_PASSAGE) -> ProductionState:
        config = _make_config(passage_text=passage, style_conflict=style_conflict)
        state = ProductionState(config=config)
        state = NarratorAgent().run(state)
        state = DirectorAgent().run(state)
        return state

    def test_no_conflict_notes_unchanged(self):
        state = self._run_director(style_conflict=None)
        for scene in state.scenes:
            assert "CONFLICT-RESOLVED" not in scene.director_notes

    def test_conflict_appended_to_notes(self):
        state = self._run_director(style_conflict=_conflict())
        for scene in state.scenes:
            assert "CONFLICT-RESOLVED" in scene.director_notes

    def test_kinetic_level_in_notes(self):
        state = self._run_director(style_conflict=_conflict())
        for scene in state.scenes:
            assert "explosive/violent" in scene.director_notes

    def test_particle_keywords_in_notes(self):
        state = self._run_director(style_conflict=_conflict())
        combined = " ".join(s.director_notes for s in state.scenes)
        assert "debris" in combined
        assert "shards" in combined
        assert "fragmentation" in combined

    def test_blur_discard_instruction_in_notes(self):
        state = self._run_director(style_conflict=_conflict())
        for scene in state.scenes:
            assert "DISCARD" in scene.director_notes

    def test_metadata_records_conflict(self):
        state = self._run_director(style_conflict=_conflict())
        assert state.metadata.get("director_style_conflict_active") is True
        assert state.metadata.get("director_kinetic_level") == "explosive/violent"

    def test_metadata_no_conflict_key_when_inactive(self):
        state = self._run_director(style_conflict=None)
        assert "director_style_conflict_active" not in state.metadata


# ---------------------------------------------------------------------------
# ValidatorAgent — unit tests
# ---------------------------------------------------------------------------


class TestValidatorAgentBlurCheck:
    def _run(self, prompt: str, conflict: bool = True) -> Shot:
        sc = _conflict() if conflict else None
        config = _make_config(style_conflict=sc)
        state = ProductionState(config=config)
        shot = _make_shot(prompt)
        shot.consistency_anchors["palette"] = "warm golden-hour"
        state.shots = [shot]
        state = ValidatorAgent().run(state)
        return state.shots[0]

    def test_blur_pass_no_blur_keywords(self):
        shot = self._run("Debris exploded upward, shards fragmented")
        assert "PASS" in shot.consistency_anchors["validation:blur_check"]

    def test_blur_fail_on_smooth(self):
        shot = self._run("smooth flowing serene motion")
        assert "FAIL" in shot.consistency_anchors["validation:blur_check"]
        assert "smooth" in shot.consistency_anchors["validation:blur_check"]

    def test_blur_check_skipped_without_conflict(self):
        shot = self._run("smooth flowing serene motion", conflict=False)
        assert "PASS" in shot.consistency_anchors["validation:blur_check"]

    def test_blur_fail_reported_in_consistency_report(self):
        sc = _conflict()
        config = _make_config(style_conflict=sc)
        state = ProductionState(config=config)
        shot = _make_shot("smooth flowing scene")
        shot.consistency_anchors["palette"] = "warm golden-hour"
        state.shots = [shot]
        state = ValidatorAgent().run(state)
        categories = [r["category"] for r in state.consistency_report]
        assert "validation" in categories
        checks = [r["check"] for r in state.consistency_report]
        assert "blur_check" in checks


class TestValidatorAgentParticleCheck:
    def _state_with_shot(self, prompt: str, conflict: bool) -> ProductionState:
        sc = _conflict() if conflict else None
        config = _make_config(style_conflict=sc)
        state = ProductionState(config=config)
        shot = _make_shot(prompt)
        shot.consistency_anchors["palette"] = "warm golden-hour"
        state.shots = [shot]
        return state

    def test_particle_pass_with_debris(self):
        state = self._state_with_shot("debris and shards fill the sky", conflict=True)
        state = ValidatorAgent().run(state)
        assert "PASS" in state.shots[0].consistency_anchors["validation:particle_check"]

    def test_particle_fail_when_conflict_and_no_particles(self):
        state = self._state_with_shot("a peaceful sunrise", conflict=True)
        state = ValidatorAgent().run(state)
        assert "FAIL" in state.shots[0].consistency_anchors["validation:particle_check"]

    def test_particle_informational_when_no_conflict(self):
        state = self._state_with_shot("a peaceful sunrise", conflict=False)
        state = ValidatorAgent().run(state)
        # No conflict → particle check should still PASS (not enforced)
        assert "PASS" in state.shots[0].consistency_anchors["validation:particle_check"]

    def test_particle_fail_reported_in_consistency_report(self):
        state = self._state_with_shot("a peaceful sunrise", conflict=True)
        state = ValidatorAgent().run(state)
        checks = [r["check"] for r in state.consistency_report]
        assert "particle_check" in checks

    def test_strict_particle_check_enforces_without_conflict(self):
        config = _make_config(style_conflict=None)
        state = ProductionState(config=config)
        shot = _make_shot("a peaceful sunrise")
        shot.consistency_anchors["palette"] = "warm golden-hour"
        state.shots = [shot]
        state = ValidatorAgent(strict_particle_check=True).run(state)
        assert "FAIL" in state.shots[0].consistency_anchors["validation:particle_check"]


class TestValidatorAgentConsistencyCheck:
    def test_consistency_pass_when_palette_present(self):
        config = _make_config(style_conflict=None)
        state = ProductionState(config=config)
        shot = _make_shot("warm golden-hour light fills the frame")
        shot.consistency_anchors["palette"] = "warm golden-hour lighting"
        state.shots = [shot]
        state = ValidatorAgent().run(state)
        assert "PASS" in state.shots[0].consistency_anchors["validation:consistency_check"]

    def test_consistency_pass_when_no_palette_anchor(self):
        config = _make_config(style_conflict=None)
        state = ProductionState(config=config)
        shot = _make_shot("a scene with no palette info")
        state.shots = [shot]
        state = ValidatorAgent().run(state)
        # No palette anchor → trivially passes (nothing to verify)
        assert "PASS" in state.shots[0].consistency_anchors["validation:consistency_check"]

    def test_consistency_fail_when_palette_missing_from_prompt(self):
        config = _make_config(style_conflict=None)
        state = ProductionState(config=config)
        shot = _make_shot("totally different scene content")
        shot.consistency_anchors["palette"] = "warm golden-hour lighting"
        state.shots = [shot]
        state = ValidatorAgent().run(state)
        assert "FAIL" in state.shots[0].consistency_anchors["validation:consistency_check"]


class TestValidatorAgentMetadata:
    def test_metadata_shots_checked(self):
        config = _make_config(style_conflict=None)
        state = ProductionState(config=config)
        state.shots = [_make_shot(f"scene {i}") for i in range(3)]
        state = ValidatorAgent().run(state)
        assert state.metadata["validator_shots_checked"] == 3

    def test_metadata_failure_count(self):
        config = _make_config(style_conflict=_conflict())
        state = ProductionState(config=config)
        # "smooth" triggers blur fail + no particles → particle fail
        shot = _make_shot("smooth flowing serene scene")
        shot.consistency_anchors["palette"] = "warm golden-hour lighting"
        state.shots = [shot]
        state = ValidatorAgent().run(state)
        assert state.metadata["validator_failures"] >= 2

    def test_existing_consistency_report_preserved(self):
        config = _make_config(style_conflict=None)
        state = ProductionState(config=config)
        state.consistency_report = [{"category": "pre-existing"}]
        state.shots = [_make_shot("a quiet dawn")]
        state = ValidatorAgent().run(state)
        categories = [r["category"] for r in state.consistency_report]
        assert "pre-existing" in categories


# ---------------------------------------------------------------------------
# End-to-end: PipelineOrchestrator with style_conflict
# ---------------------------------------------------------------------------


class TestOrchestratorWithStyleConflict:
    def test_pipeline_runs_with_conflict(self, tmp_path):
        config = _make_config(
            output_path=str(tmp_path),
            style_conflict=_conflict(),
        )
        state = PipelineOrchestrator(config).run()
        assert len(state.shots) > 0

    def test_validator_metadata_present(self, tmp_path):
        config = _make_config(
            output_path=str(tmp_path),
            style_conflict=_conflict(),
        )
        state = PipelineOrchestrator(config).run()
        assert "validator_shots_checked" in state.metadata
        assert state.metadata["validator_shots_checked"] == len(state.shots)

    def test_validation_anchors_in_shots(self, tmp_path):
        config = _make_config(
            output_path=str(tmp_path),
            style_conflict=_conflict(),
        )
        state = PipelineOrchestrator(config).run()
        for shot in state.shots:
            assert "validation:blur_check" in shot.consistency_anchors
            assert "validation:particle_check" in shot.consistency_anchors
            assert "validation:consistency_check" in shot.consistency_anchors

    def test_particle_keywords_propagate_to_shots(self, tmp_path):
        """Conflict injection via DirectorAgent must reach final shot prompts."""
        config = _make_config(
            output_path=str(tmp_path),
            style_conflict=_conflict(),
        )
        state = PipelineOrchestrator(config).run()
        combined = " ".join(sh.prompt for sh in state.shots)
        assert any(kw in combined for kw in ["debris", "shards", "fragmentation"])

    def test_no_conflict_validator_still_runs(self, tmp_path):
        config = _make_config(output_path=str(tmp_path), style_conflict=None)
        state = PipelineOrchestrator(config).run()
        assert "validator_shots_checked" in state.metadata

    def test_director_metadata_conflict_flag(self, tmp_path):
        config = _make_config(
            output_path=str(tmp_path),
            style_conflict=_conflict(),
        )
        state = PipelineOrchestrator(config).run()
        assert state.metadata.get("director_style_conflict_active") is True

    def test_from_yaml_parses_style_conflict(self, tmp_path):
        yaml_content = textwrap.dedent(f"""\
            pipeline:
              name: "Action-Genesis"
              style: cinematic

            passage:
              text: |
                Continental plates ruptured and mountains shattered into the sky.
                Debris and fragmentation filled the atmosphere.

            characters: []

            style_conflict:
              kinetic_level: "explosive/violent"
              shutter: "1/1000"
              physics_override:
                fluid_turbulence: "chaotic high-pressure jets"

            consistency:
              palette: storm_grey

            output:
              format: json
              path: {tmp_path}
        """)
        cfg_path = tmp_path / "action_genesis.yaml"
        cfg_path.write_text(yaml_content)
        state = PipelineOrchestrator.from_yaml(cfg_path).run()
        assert state.config.style_conflict is not None
        assert state.config.style_conflict.kinetic_level == "explosive/violent"
        assert state.config.style_conflict.shutter == "1/1000"
        assert "fluid_turbulence" in state.config.style_conflict.physics_override
        assert "validator_shots_checked" in state.metadata

    def test_from_yaml_missing_kinetic_level_ignored(self, tmp_path):
        yaml_content = textwrap.dedent(f"""\
            pipeline:
              name: "Bad Conflict"
              style: cinematic

            passage:
              text: |
                The earth was quiet in the morning.

            characters: []

            style_conflict:
              shutter: "1/500"

            consistency:
              palette: warm_golden_dawn

            output:
              format: json
              path: {tmp_path}
        """)
        cfg_path = tmp_path / "bad_conflict.yaml"
        cfg_path.write_text(yaml_content)
        # Should not raise — missing kinetic_level is logged and ignored
        state = PipelineOrchestrator.from_yaml(cfg_path).run()
        assert state.config.style_conflict is None
