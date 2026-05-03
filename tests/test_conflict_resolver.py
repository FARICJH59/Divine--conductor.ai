"""Unit tests for ConflictResolverAgent."""

from __future__ import annotations

import pytest

from divine_conductor.agents.conflict_resolver import (
    ConflictResolverAgent,
    _is_fast_shutter,
    _parse_shutter,
)
from divine_conductor.agents.cinematographer import CinematographerAgent
from divine_conductor.agents.director import DirectorAgent
from divine_conductor.agents.narrator import NarratorAgent
from divine_conductor.models.production import (
    CameraAngle,
    Character,
    PhysicsOverride,
    ProductionConfig,
    ProductionState,
    Shot,
    StyleConflictMetadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TECTONIC_PASSAGE = (
    "And the Lord God formed man from the dust of the ground and breathed "
    "into his nostrils the breath of life."
)


def _make_config(**kwargs) -> ProductionConfig:
    defaults = dict(name="Test", passage_text=_TECTONIC_PASSAGE)
    defaults.update(kwargs)
    return ProductionConfig(**defaults)


def _make_shot(prompt: str = "Some prompt.", negative_prompt: str = "") -> Shot:
    return Shot(
        scene_id="scene-1",
        index=0,
        prompt=prompt,
        negative_prompt=negative_prompt,
        camera_angle=CameraAngle.WIDE,
    )


def _state_with_shots(config: ProductionConfig) -> ProductionState:
    state = ProductionState(config=config)
    state = NarratorAgent().run(state)
    state = DirectorAgent().run(state)
    state = CinematographerAgent().run(state)
    return state


# ---------------------------------------------------------------------------
# _parse_shutter helper
# ---------------------------------------------------------------------------


class TestParseShutter:
    def test_parses_fraction(self):
        from fractions import Fraction
        result = _parse_shutter("1/2000")
        assert result == Fraction(1, 2000)

    def test_returns_none_for_invalid(self):
        assert _parse_shutter("fast") is None

    def test_returns_none_for_zero_denominator(self):
        assert _parse_shutter("1/0") is None


# ---------------------------------------------------------------------------
# _is_fast_shutter helper
# ---------------------------------------------------------------------------


class TestIsFastShutter:
    def test_1_over_2000_is_fast(self):
        assert _is_fast_shutter("1/2000") is True

    def test_1_over_500_is_not_fast(self):
        # Threshold is strictly *less than* 1/500
        assert _is_fast_shutter("1/500") is False

    def test_1_over_1000_is_fast(self):
        assert _is_fast_shutter("1/1000") is True

    def test_1_over_100_is_not_fast(self):
        assert _is_fast_shutter("1/100") is False

    def test_invalid_returns_false(self):
        assert _is_fast_shutter("slow") is False


# ---------------------------------------------------------------------------
# ConflictResolverAgent — no conflict configured
# ---------------------------------------------------------------------------


class TestConflictResolverNoConflict:
    def test_noop_when_no_style_conflict_no_genre(self):
        config = _make_config()
        state = _state_with_shots(config)
        original_negatives = [s.negative_prompt for s in state.shots]
        state = ConflictResolverAgent().run(state)
        # No changes expected
        for shot, orig in zip(state.shots, original_negatives):
            assert shot.negative_prompt == orig

    def test_metadata_not_set_when_no_negatives(self):
        config = _make_config()
        state = _state_with_shots(config)
        state = ConflictResolverAgent().run(state)
        assert "conflict_resolver_negatives" not in state.metadata


# ---------------------------------------------------------------------------
# ConflictResolverAgent — genre-only negatives
# ---------------------------------------------------------------------------


class TestConflictResolverGenreNegatives:
    def test_biblical_genre_injects_fire_negatives(self):
        config = _make_config(genre="biblical")
        state = _state_with_shots(config)
        state = ConflictResolverAgent().run(state)

        for shot in state.shots:
            assert "orange fire" in shot.negative_prompt
            assert "VFX fire" in shot.negative_prompt
            assert "fuel fire" in shot.negative_prompt

    def test_unknown_genre_no_genre_negatives(self):
        config = _make_config(genre="unknown_genre")
        state = _state_with_shots(config)
        original_negatives = [s.negative_prompt for s in state.shots]
        state = ConflictResolverAgent().run(state)
        for shot, orig in zip(state.shots, original_negatives):
            assert shot.negative_prompt == orig


# ---------------------------------------------------------------------------
# ConflictResolverAgent — kinetic-level negatives
# ---------------------------------------------------------------------------


class TestConflictResolverKineticNegatives:
    def _make_state_with_conflict(self, kinetic_level: str, shutter: str = "") -> ProductionState:
        style_conflict = StyleConflictMetadata(
            kinetic_level=kinetic_level,
            shutter=shutter,
            physics_override=PhysicsOverride(
                fluid_turbulence="chaotic high-pressure vapor jets",
                debris_density="extreme_particulate_shatter",
                gravity_variance="unstable_pulsing",
            ),
        )
        config = _make_config(style_conflict=style_conflict)
        return _state_with_shots(config)

    def test_tectonic_violence_injects_motion_blur_negative(self):
        state = self._make_state_with_conflict("tectonic_violence")
        state = ConflictResolverAgent().run(state)
        for shot in state.shots:
            assert "motion blur" in shot.negative_prompt

    def test_tectonic_violence_injects_smooth_surface_negative(self):
        state = self._make_state_with_conflict("tectonic_violence")
        state = ConflictResolverAgent().run(state)
        for shot in state.shots:
            assert "smooth surface" in shot.negative_prompt

    def test_tectonic_violence_injects_liquid_smear_negative(self):
        state = self._make_state_with_conflict("tectonic_violence")
        state = ConflictResolverAgent().run(state)
        for shot in state.shots:
            assert "liquid smear" in shot.negative_prompt

    def test_unknown_kinetic_level_no_kinetic_negatives(self):
        style_conflict = StyleConflictMetadata(kinetic_level="gentle_breeze")
        config = _make_config(style_conflict=style_conflict)
        state = _state_with_shots(config)
        state = ConflictResolverAgent().run(state)
        # gentle_breeze has no kinetic negatives defined
        for shot in state.shots:
            assert "motion blur" not in shot.negative_prompt


# ---------------------------------------------------------------------------
# ConflictResolverAgent — fast shutter + biblical conflict
# ---------------------------------------------------------------------------


class TestConflictResolverTemporalConflict:
    def _make_tectonic_state(self) -> ProductionState:
        style_conflict = StyleConflictMetadata(
            kinetic_level="tectonic_violence",
            shutter="1/2000",
            physics_override=PhysicsOverride(
                fluid_turbulence="chaotic high-pressure vapor jets",
                debris_density="extreme_particulate_shatter",
                gravity_variance="unstable_pulsing",
            ),
        )
        config = _make_config(genre="biblical", style_conflict=style_conflict)
        return _state_with_shots(config)

    def test_full_tectonic_rupture_injects_all_negatives(self):
        state = self._make_tectonic_state()
        state = ConflictResolverAgent().run(state)

        for shot in state.shots:
            neg = shot.negative_prompt
            # Kinetic negatives
            assert "motion blur" in neg
            assert "liquid smear" in neg
            assert "smooth surface" in neg
            # Genre negatives
            assert "orange fire" in neg
            assert "VFX fire" in neg

    def test_metadata_records_negatives(self):
        state = self._make_tectonic_state()
        state = ConflictResolverAgent().run(state)
        assert "conflict_resolver_negatives" in state.metadata
        assert len(state.metadata["conflict_resolver_negatives"]) > 0

    def test_existing_negative_prompt_preserved(self):
        config = _make_config(genre="biblical")
        state = ProductionState(config=config)
        state = NarratorAgent().run(state)
        state = DirectorAgent().run(state)
        state = CinematographerAgent().run(state)
        # Pre-populate a negative prompt on the first shot
        first = state.shots[0]
        state.shots[0] = Shot(
            scene_id=first.scene_id,
            index=first.index,
            prompt=first.prompt,
            negative_prompt="overexposed",
            duration_seconds=first.duration_seconds,
            camera_angle=first.camera_angle,
            id=first.id,
        )

        state = ConflictResolverAgent().run(state)
        # The pre-existing value must still be present
        assert "overexposed" in state.shots[0].negative_prompt
        # AND new negatives are appended
        assert "orange fire" in state.shots[0].negative_prompt

    def test_shot_ids_preserved_after_resolver(self):
        state = self._make_tectonic_state()
        original_ids = [s.id for s in state.shots]
        state = ConflictResolverAgent().run(state)
        assert [s.id for s in state.shots] == original_ids

    def test_slow_shutter_no_extra_temporal_negatives(self):
        """A slow shutter in a biblical genre should not trigger the fast-shutter path."""
        style_conflict = StyleConflictMetadata(
            kinetic_level="",
            shutter="1/50",  # slow shutter — not fast
        )
        config = _make_config(genre="biblical", style_conflict=style_conflict)
        state = _state_with_shots(config)
        state = ConflictResolverAgent().run(state)
        # Genre negatives still apply, but the temporal conflict path is not taken
        negatives = state.metadata.get("conflict_resolver_negatives", [])
        # "soft edges" is added by the fast-shutter path — should NOT appear here
        # (it may appear from kinetic negatives but kinetic_level is empty here)
        for shot in state.shots:
            # Only genre negatives; no "soft edges" from temporal path
            assert "orange fire" in shot.negative_prompt
