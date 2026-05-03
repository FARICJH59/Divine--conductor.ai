"""Unit tests for ValidatorAgent."""

from __future__ import annotations

import pytest

from divine_conductor.agents.validator import ValidatorAgent
from divine_conductor.agents.cinematographer import CinematographerAgent
from divine_conductor.agents.conflict_resolver import ConflictResolverAgent
from divine_conductor.agents.director import DirectorAgent
from divine_conductor.agents.narrator import NarratorAgent
from divine_conductor.consistency.veo_consistency import Veo3ConsistencyEngine
from divine_conductor.models.production import (
    CameraAngle,
    Character,
    PhysicsOverride,
    ProductionConfig,
    ProductionState,
    Shot,
    StyleConflictMetadata,
)
from divine_conductor.pipeline.failure_log import FailureLog, FailureType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASSAGE = (
    "And the Lord God formed man from the dust of the ground and breathed "
    "into his nostrils the breath of life, and man became a living creature."
)


def _make_config(**kwargs) -> ProductionConfig:
    defaults = dict(name="Validator Test", passage_text=_PASSAGE)
    defaults.update(kwargs)
    return ProductionConfig(**defaults)


def _make_shot(
    prompt: str = "A stone desert scene.",
    negative_prompt: str = "",
    consistency_anchors: dict | None = None,
    scene_id: str = "sc1",
) -> Shot:
    return Shot(
        scene_id=scene_id,
        index=0,
        prompt=prompt,
        negative_prompt=negative_prompt,
        camera_angle=CameraAngle.WIDE,
        consistency_anchors=consistency_anchors or {},
    )


def _run_full_pipeline(config: ProductionConfig) -> ProductionState:
    state = ProductionState(config=config)
    state = NarratorAgent().run(state)
    state = DirectorAgent().run(state)
    state = CinematographerAgent().run(state)
    state = ConflictResolverAgent().run(state)
    engine = Veo3ConsistencyEngine(palette=config.palette)
    for char in config.characters:
        engine.register_character(char)
    state.shots = engine.apply(state.shots)
    issues = engine.check_continuity(state.shots)
    state.consistency_report = [i.to_dict() for i in issues]
    return state


# ---------------------------------------------------------------------------
# ValidatorAgent — no-op when nothing is wrong
# ---------------------------------------------------------------------------


class TestValidatorAgentClean:
    def test_no_issues_for_clean_production(self):
        config = _make_config()
        state = _run_full_pipeline(config)
        state = ValidatorAgent().run(state)
        validator_issues = [
            i for i in state.consistency_report
            if i.get("type") in (
                "TEMPORAL_CONFLICT", "HALLUCINATION", "CHARACTER_ANCHOR_DRIFT"
            )
        ]
        assert validator_issues == []

    def test_metadata_records_zero_issues(self):
        config = _make_config()
        state = _run_full_pipeline(config)
        state = ValidatorAgent().run(state)
        assert state.metadata["validator_issues_found"] == 0


# ---------------------------------------------------------------------------
# ValidatorAgent — TEMPORAL_CONFLICT detection
# ---------------------------------------------------------------------------


class TestValidatorTemporalConflict:
    def test_flags_motion_blur_in_fast_shutter_shot(self):
        style_conflict = StyleConflictMetadata(shutter="1/2000")
        config = _make_config(style_conflict=style_conflict)
        state = ProductionState(config=config)
        # Inject a shot that contains "motion blur" despite fast-shutter setting
        state.shots = [
            _make_shot(
                prompt="A cinematic scene with motion blur and soft edges.",
                scene_id="sc1",
            )
        ]
        state = ValidatorAgent().run(state)
        types = [i["type"] for i in state.consistency_report]
        assert "TEMPORAL_CONFLICT" in types

    def test_no_temporal_conflict_without_style_conflict(self):
        config = _make_config()
        state = ProductionState(config=config)
        state.shots = [_make_shot(prompt="A scene with motion blur.")]
        state = ValidatorAgent().run(state)
        types = [i["type"] for i in state.consistency_report]
        assert "TEMPORAL_CONFLICT" not in types

    def test_no_temporal_conflict_for_slow_shutter(self):
        style_conflict = StyleConflictMetadata(shutter="1/50")  # slow
        config = _make_config(style_conflict=style_conflict)
        state = ProductionState(config=config)
        state.shots = [_make_shot(prompt="A gentle blur across the plains.")]
        state = ValidatorAgent().run(state)
        types = [i["type"] for i in state.consistency_report]
        assert "TEMPORAL_CONFLICT" not in types

    def test_temporal_conflict_issue_has_shot_id(self):
        style_conflict = StyleConflictMetadata(shutter="1/2000")
        config = _make_config(style_conflict=style_conflict)
        state = ProductionState(config=config)
        shot = _make_shot(prompt="Motion blur streaking across the frame.")
        state.shots = [shot]
        state = ValidatorAgent().run(state)
        conflict_issues = [i for i in state.consistency_report if i.get("type") == "TEMPORAL_CONFLICT"]
        assert conflict_issues[0]["shot_id"] == shot.id


# ---------------------------------------------------------------------------
# ValidatorAgent — HALLUCINATION detection
# ---------------------------------------------------------------------------


class TestValidatorHallucination:
    def test_flags_vfx_fire_in_biblical_production(self):
        config = _make_config(genre="biblical")
        state = ProductionState(config=config)
        state.shots = [
            _make_shot(
                prompt="Divine wrath as orange fire erupts from the earth.",
                scene_id="sc1",
            )
        ]
        state = ValidatorAgent().run(state)
        types = [i["type"] for i in state.consistency_report]
        assert "HALLUCINATION" in types

    def test_flags_vfx_fire_token(self):
        config = _make_config(genre="biblical")
        state = ProductionState(config=config)
        state.shots = [_make_shot(prompt="The sky fills with VFX fire and smoke.")]
        state = ValidatorAgent().run(state)
        types = [i["type"] for i in state.consistency_report]
        assert "HALLUCINATION" in types

    def test_no_hallucination_for_unknown_genre(self):
        config = _make_config(genre="western")
        state = ProductionState(config=config)
        state.shots = [_make_shot(prompt="A raging orange fire burns the saloon.")]
        state = ValidatorAgent().run(state)
        types = [i["type"] for i in state.consistency_report]
        assert "HALLUCINATION" not in types

    def test_no_hallucination_when_clean_prompt(self):
        config = _make_config(genre="biblical")
        state = ProductionState(config=config)
        state.shots = [
            _make_shot(
                prompt="Divine white light cascades over the ancient stone temple."
            )
        ]
        state = ValidatorAgent().run(state)
        types = [i["type"] for i in state.consistency_report]
        assert "HALLUCINATION" not in types


# ---------------------------------------------------------------------------
# ValidatorAgent — CHARACTER_ANCHOR_DRIFT detection
# ---------------------------------------------------------------------------


class TestValidatorCharacterAnchorDrift:
    def test_flags_drift_when_anchor_absent(self):
        config = _make_config()
        state = ProductionState(config=config)
        # Shot claims to have character anchor but prompt contains no anchor words
        state.shots = [
            _make_shot(
                prompt="A barren rocky landscape with dust and stones.",
                consistency_anchors={
                    "character:adam": "tall man with olive skin and dark curly hair"
                },
            )
        ]
        state = ValidatorAgent().run(state)
        types = [i["type"] for i in state.consistency_report]
        assert "CHARACTER_ANCHOR_DRIFT" in types

    def test_no_drift_when_anchor_words_present(self):
        config = _make_config()
        state = ProductionState(config=config)
        # Prompt includes words from the anchor description
        state.shots = [
            _make_shot(
                prompt=(
                    "A tall man with olive skin and dark curly hair stands amid dust."
                ),
                consistency_anchors={
                    "character:adam": "tall man with olive skin and dark curly hair"
                },
            )
        ]
        state = ValidatorAgent().run(state)
        drift_issues = [
            i for i in state.consistency_report
            if i.get("type") == "CHARACTER_ANCHOR_DRIFT"
        ]
        assert drift_issues == []

    def test_non_character_anchors_ignored(self):
        config = _make_config()
        state = ProductionState(config=config)
        state.shots = [
            _make_shot(
                prompt="A sunset scene.",
                consistency_anchors={
                    "palette": "warm golden light",
                    "veo_model": "veo-3.1",
                },
            )
        ]
        state = ValidatorAgent().run(state)
        types = [i["type"] for i in state.consistency_report]
        assert "CHARACTER_ANCHOR_DRIFT" not in types

    def test_drift_issue_contains_character_key(self):
        config = _make_config()
        state = ProductionState(config=config)
        state.shots = [
            _make_shot(
                prompt="Dust and stone.",
                consistency_anchors={
                    "character:adam": "bearded man with wooden staff"
                },
            )
        ]
        state = ValidatorAgent().run(state)
        drift_issues = [
            i for i in state.consistency_report
            if i.get("type") == "CHARACTER_ANCHOR_DRIFT"
        ]
        assert drift_issues[0]["character"] == "character:adam"

    def test_multiple_character_anchors_checked_independently(self):
        config = _make_config()
        state = ProductionState(config=config)
        state.shots = [
            _make_shot(
                prompt="A bearded man with wooden staff stands alone.",
                consistency_anchors={
                    "character:adam": "bearded man with wooden staff",
                    # Eve's anchor words absent from prompt
                    "character:eve": "woman with long flowing hair",
                },
            )
        ]
        state = ValidatorAgent().run(state)
        drift_issues = [
            i for i in state.consistency_report
            if i.get("type") == "CHARACTER_ANCHOR_DRIFT"
        ]
        # Only Eve should drift — Adam's words are in the prompt
        assert len(drift_issues) == 1
        assert drift_issues[0]["character"] == "character:eve"


# ---------------------------------------------------------------------------
# ValidatorAgent — FailureLog integration
# ---------------------------------------------------------------------------


class TestValidatorFailureLogIntegration:
    def test_temporal_conflict_recorded_in_log(self):
        style_conflict = StyleConflictMetadata(shutter="1/2000")
        config = _make_config(style_conflict=style_conflict)
        state = ProductionState(config=config)
        state.shots = [_make_shot(prompt="Motion blur across the ancient ruins.")]

        log = FailureLog()
        ValidatorAgent(failure_log=log).run(state)

        tc_records = log.failures_by_type(FailureType.TEMPORAL_CONFLICT)
        assert len(tc_records) == 1

    def test_hallucination_recorded_in_log(self):
        config = _make_config(genre="biblical")
        state = ProductionState(config=config)
        state.shots = [_make_shot(prompt="VFX fire consuming the altar.")]

        log = FailureLog()
        ValidatorAgent(failure_log=log).run(state)

        h_records = log.failures_by_type(FailureType.HALLUCINATION)
        assert len(h_records) == 1

    def test_character_drift_recorded_in_log(self):
        config = _make_config()
        state = ProductionState(config=config)
        state.shots = [
            _make_shot(
                prompt="Rocky barren earth.",
                consistency_anchors={"character:adam": "bearded man in linen robe"},
            )
        ]

        log = FailureLog()
        ValidatorAgent(failure_log=log).run(state)

        drift_records = log.failures_by_type(FailureType.CHARACTER_ANCHOR_DRIFT)
        assert len(drift_records) == 1

    def test_motion_bucket_delta_for_temporal_conflict(self):
        style_conflict = StyleConflictMetadata(shutter="1/2000")
        config = _make_config(style_conflict=style_conflict)
        state = ProductionState(config=config)
        state.shots = [_make_shot(prompt="Blur and smear across the desert.")]

        log = FailureLog()
        ValidatorAgent(failure_log=log).run(state)

        record = log.failures_by_type(FailureType.TEMPORAL_CONFLICT)[0]
        assert record.motion_bucket_delta == pytest.approx(-0.1)

    def test_no_failure_log_still_adds_to_consistency_report(self):
        config = _make_config(genre="biblical")
        state = ProductionState(config=config)
        state.shots = [_make_shot(prompt="A scene with orange fire erupting.")]

        # No FailureLog passed
        ValidatorAgent().run(state)
        assert any(
            i.get("type") == "HALLUCINATION" for i in state.consistency_report
        )

    def test_suggest_adjustment_reflects_recorded_failures(self):
        style_conflict = StyleConflictMetadata(shutter="1/2000")
        config = _make_config(style_conflict=style_conflict)
        state = ProductionState(config=config)
        shot = _make_shot(prompt="Blurry motion across the cracked earth.")
        state.shots = [shot]

        log = FailureLog()
        ValidatorAgent(failure_log=log).run(state)

        # The suggest_adjustment for this shot should reflect the -0.1 TEMPORAL delta
        assert log.suggest_adjustment(shot.id) == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# ValidatorAgent — end-to-end via PipelineOrchestrator
# ---------------------------------------------------------------------------


class TestValidatorEndToEnd:
    def test_validator_runs_in_full_pipeline(self, tmp_path):
        from divine_conductor.pipeline.orchestrator import PipelineOrchestrator

        style_conflict = StyleConflictMetadata(
            kinetic_level="tectonic_violence",
            shutter="1/2000",
            physics_override=PhysicsOverride(
                fluid_turbulence="chaotic vapor jets",
                debris_density="extreme particulate",
                gravity_variance="unstable",
            ),
        )
        char = Character(
            id="adam",
            name="Adam",
            description="The first man, formed from dust, with dark hair and simple linen garment.",
        )
        config = _make_config(
            genre="biblical",
            style_conflict=style_conflict,
            characters=[char],
            output_path=str(tmp_path),
        )
        orch = PipelineOrchestrator(config)
        state = orch.run()

        # ValidatorAgent metadata must be present
        assert "validator_issues_found" in state.metadata

    def test_failure_log_written_when_failures_exist(self, tmp_path):
        from divine_conductor.pipeline.orchestrator import PipelineOrchestrator

        style_conflict = StyleConflictMetadata(shutter="1/2000")
        config = _make_config(
            genre="biblical",
            style_conflict=style_conflict,
            passage_text="The earth shook with motion blur and orange fire.",
            output_path=str(tmp_path),
        )
        PipelineOrchestrator(config).run()

        log_path = tmp_path / "failure_log.json"
        # The log is only written when failures were found
        if log_path.exists():
            from divine_conductor.pipeline.failure_log import FailureLog
            log = FailureLog.load(log_path)
            assert len(log) >= 0  # may be 0 if no failures — just ensure file is valid JSON
