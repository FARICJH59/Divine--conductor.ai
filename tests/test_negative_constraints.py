"""Tests for the NegativeConstraintManager."""

import pytest

from divine_conductor.consistency.negative_constraints import (
    GENRE_NEGATIVES,
    NegativeConstraintManager,
)


class TestNegativeConstraintManagerInit:
    def test_known_genre_loads_forbidden_list(self):
        mgr = NegativeConstraintManager("biblical")
        assert mgr.forbidden_list == GENRE_NEGATIVES["biblical"]

    def test_genre_lookup_is_case_insensitive(self):
        mgr_lower = NegativeConstraintManager("biblical")
        mgr_upper = NegativeConstraintManager("BIBLICAL")
        assert mgr_lower.forbidden_list == mgr_upper.forbidden_list

    def test_unknown_genre_gives_empty_forbidden_list(self):
        mgr = NegativeConstraintManager("fantasy")
        assert mgr.forbidden_list == []

    def test_empty_string_genre_gives_empty_forbidden_list(self):
        mgr = NegativeConstraintManager("")
        assert mgr.forbidden_list == []

    def test_all_defined_genres_load(self):
        for genre in GENRE_NEGATIVES:
            mgr = NegativeConstraintManager(genre)
            assert mgr.forbidden_list == GENRE_NEGATIVES[genre]


class TestGetNegativePrompt:
    def test_output_is_string(self):
        mgr = NegativeConstraintManager("biblical")
        result = mgr.get_negative_prompt()
        assert isinstance(result, str)

    def test_known_genre_contains_forbidden_terms(self):
        mgr = NegativeConstraintManager("biblical")
        result = mgr.get_negative_prompt()
        for term in GENRE_NEGATIVES["biblical"]:
            assert term in result

    def test_quality_guard_always_appended(self):
        for genre in ("biblical", "cyber_noir", "action_kinetic", "unknown", ""):
            mgr = NegativeConstraintManager(genre)
            result = mgr.get_negative_prompt()
            assert "low quality" in result
            assert "distorted anatomy" in result
            assert "glitches" in result

    def test_unknown_genre_returns_only_quality_guard(self):
        mgr = NegativeConstraintManager("unknown_genre")
        result = mgr.get_negative_prompt()
        # Should be exactly the quality guard with no leading comma or space
        assert result == "low quality, distorted anatomy, glitches"

    def test_terms_are_comma_separated(self):
        mgr = NegativeConstraintManager("biblical")
        result = mgr.get_negative_prompt()
        # Every term from the forbidden list should be findable as a segment
        parts = [p.strip() for p in result.split(",")]
        for term in GENRE_NEGATIVES["biblical"]:
            assert term in parts

    def test_cyber_noir_genre(self):
        mgr = NegativeConstraintManager("cyber_noir")
        result = mgr.get_negative_prompt()
        assert "natural bright sunlight" in result
        assert "optimism" in result

    def test_action_kinetic_genre(self):
        mgr = NegativeConstraintManager("action_kinetic")
        result = mgr.get_negative_prompt()
        assert "motion blur" in result
        assert "serenity" in result


class TestCinematographerIntegration:
    """Verify that CinematographerAgent populates Shot.negative_prompt via the manager."""

    def test_shots_carry_negative_prompt_for_biblical_genre(self):
        from divine_conductor.agents.cinematographer import CinematographerAgent
        from divine_conductor.agents.director import DirectorAgent
        from divine_conductor.agents.narrator import NarratorAgent
        from divine_conductor.models.production import ProductionConfig, ProductionState

        config = ProductionConfig(
            name="Negative Prompt Test",
            passage_text="God created the heavens and the earth.",
            genre="biblical",
        )
        state = ProductionState(config=config)
        state = NarratorAgent().run(state)
        state = DirectorAgent().run(state)
        state = CinematographerAgent().run(state)

        assert len(state.shots) > 0
        for shot in state.shots:
            assert "plastic" in shot.negative_prompt
            assert "low quality" in shot.negative_prompt

    def test_shots_negative_prompt_empty_genre_has_only_quality_guard(self):
        from divine_conductor.agents.cinematographer import CinematographerAgent
        from divine_conductor.agents.director import DirectorAgent
        from divine_conductor.agents.narrator import NarratorAgent
        from divine_conductor.models.production import ProductionConfig, ProductionState

        config = ProductionConfig(
            name="No Genre Test",
            passage_text="The stars filled the night sky.",
            genre="",
        )
        state = ProductionState(config=config)
        state = NarratorAgent().run(state)
        state = DirectorAgent().run(state)
        state = CinematographerAgent().run(state)

        assert len(state.shots) > 0
        for shot in state.shots:
            assert shot.negative_prompt == "low quality, distorted anatomy, glitches"
