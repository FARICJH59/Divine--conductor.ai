"""Tests for GenreFactory and genre DNA integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from divine_conductor.core.factory import GenreFactory
from divine_conductor.agents.director import DirectorAgent
from divine_conductor.agents.cinematographer import CinematographerAgent
from divine_conductor.agents.narrator import NarratorAgent
from divine_conductor.models.production import ProductionConfig, ProductionState
from divine_conductor.pipeline.orchestrator import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASSAGE = (
    "In the beginning God created the heavens and the earth. "
    "And God said, let there be light, and there was light."
)


def _make_config(**kwargs) -> ProductionConfig:
    defaults = dict(name="Test", passage_text=_PASSAGE)
    defaults.update(kwargs)
    return ProductionConfig(**defaults)


def _genres_dir() -> Path:
    """Resolve the real config/genres/ directory from the repo root."""
    return Path(__file__).parent.parent / "config" / "genres"


# ---------------------------------------------------------------------------
# GenreFactory — registry loading
# ---------------------------------------------------------------------------


class TestGenreFactory:
    def test_loads_all_yaml_files(self):
        factory = GenreFactory(_genres_dir())
        expected = {"biblical", "sci_fi", "noir", "action"}
        assert expected.issubset(set(factory.available_genres))

    def test_available_genres_sorted(self):
        factory = GenreFactory(_genres_dir())
        genres = factory.available_genres
        assert genres == sorted(genres)

    def test_get_known_genre(self):
        factory = GenreFactory(_genres_dir())
        dna = factory.get_genre_dna("biblical")
        assert dna["genre"] == "biblical"
        assert "vibe" in dna
        assert "lighting" in dna
        assert "camera_tech" in dna

    def test_get_unknown_genre_falls_back_to_biblical(self):
        factory = GenreFactory(_genres_dir())
        dna = factory.get_genre_dna("nonexistent_genre")
        assert dna["genre"] == "biblical"

    def test_get_sci_fi_genre(self):
        factory = GenreFactory(_genres_dir())
        dna = factory.get_genre_dna("sci_fi")
        assert dna["genre"] == "sci_fi"
        assert dna["vibe"]
        assert dna["lighting"]
        assert dna["camera_tech"]

    def test_missing_directory_returns_empty_registry(self, tmp_path):
        factory = GenreFactory(tmp_path / "nonexistent")
        assert factory.available_genres == []

    def test_custom_genre_directory(self, tmp_path):
        genre_data = {"genre": "western", "vibe": "dusty", "lighting": "harsh sun", "camera_tech": "Super 35"}
        (tmp_path / "western.yaml").write_text(yaml.dump(genre_data), encoding="utf-8")
        factory = GenreFactory(tmp_path)
        assert "western" in factory.available_genres
        dna = factory.get_genre_dna("western")
        assert dna["genre"] == "western"

    def test_non_yaml_files_ignored(self, tmp_path):
        (tmp_path / "readme.txt").write_text("ignore me")
        (tmp_path / "genre.yaml").write_text(yaml.dump({"genre": "test", "vibe": "v", "lighting": "l", "camera_tech": "c"}))
        factory = GenreFactory(tmp_path)
        assert factory.available_genres == ["genre"]

    def test_invalid_yaml_mapping_skipped(self, tmp_path):
        (tmp_path / "bad.yaml").write_text("- item1\n- item2\n")
        factory = GenreFactory(tmp_path)
        assert "bad" not in factory.available_genres


# ---------------------------------------------------------------------------
# DirectorAgent — genre DNA integration
# ---------------------------------------------------------------------------


class TestDirectorAgentWithGenre:
    def _state(self) -> ProductionState:
        state = ProductionState(config=_make_config())
        return NarratorAgent().run(state)

    def test_genre_dna_appended_to_notes(self):
        dna = {"genre": "noir", "vibe": "dark and brooding", "lighting": "chiaroscuro", "camera_tech": "ARRI AMIRA"}
        state = self._state()
        state = DirectorAgent(genre_dna=dna).run(state)
        for scene in state.scenes:
            assert "dark and brooding" in scene.director_notes
            assert "ARRI AMIRA" in scene.director_notes

    def test_no_genre_dna_uses_defaults(self):
        state = self._state()
        state = DirectorAgent().run(state)
        for scene in state.scenes:
            assert scene.director_notes  # still populated from templates

    def test_genre_recorded_in_metadata(self):
        dna = {"genre": "action", "vibe": "visceral", "lighting": "harsh", "camera_tech": "ALEXA Mini"}
        state = self._state()
        state = DirectorAgent(genre_dna=dna).run(state)
        assert state.metadata["director_genre"] == "action"


# ---------------------------------------------------------------------------
# CinematographerAgent — genre DNA integration
# ---------------------------------------------------------------------------


class TestCinematographerAgentWithGenre:
    def _directed_state(self) -> ProductionState:
        state = ProductionState(config=_make_config())
        state = NarratorAgent().run(state)
        state = DirectorAgent().run(state)
        return state

    def test_genre_lighting_in_shot_prompt(self):
        dna = {"genre": "sci_fi", "vibe": "futuristic", "lighting": "neon-blue practicals", "camera_tech": "ultra-wide anamorphic"}
        state = self._directed_state()
        state = CinematographerAgent(genre_dna=dna).run(state)
        combined = " ".join(sh.prompt for sh in state.shots)
        assert "neon-blue practicals" in combined
        assert "ultra-wide anamorphic" in combined

    def test_genre_recorded_in_metadata(self):
        dna = {"genre": "sci_fi", "vibe": "v", "lighting": "l", "camera_tech": "c"}
        state = self._directed_state()
        state = CinematographerAgent(genre_dna=dna).run(state)
        assert state.metadata["cinematographer_genre"] == "sci_fi"


# ---------------------------------------------------------------------------
# PipelineOrchestrator — end-to-end genre integration
# ---------------------------------------------------------------------------


class TestPipelineOrchestratorWithGenre:
    def test_genre_biblical_pipeline(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        orch = PipelineOrchestrator(config, genre="biblical")
        state = orch.run()
        assert len(state.shots) > 0
        # God rays from biblical genre should appear somewhere in prompts
        combined = " ".join(sh.prompt for sh in state.shots)
        assert "god rays" in combined.lower() or "sacred" in combined.lower()

    def test_genre_noir_pipeline(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        orch = PipelineOrchestrator(config, genre="noir")
        state = orch.run()
        combined = " ".join(sh.prompt for sh in state.shots)
        assert "chiaroscuro" in combined.lower() or "noir" in combined.lower()

    def test_no_genre_pipeline_unchanged(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        orch = PipelineOrchestrator(config, genre=None)
        state = orch.run()
        assert len(state.shots) > 0

    def test_from_yaml_with_genre_field(self, tmp_path):
        yaml_content = f"""
pipeline:
  name: "Genre YAML Test"
  style: cinematic
  genre: sci_fi

passage:
  text: |
    The starship emerged from hyperspace into a field of dying suns.

characters: []

consistency:
  palette: moonlit_blue
  anchor_shots: true
  character_id_strength: 0.8

output:
  format: json
  path: {tmp_path}
"""
        cfg_path = tmp_path / "genre_test.yaml"
        cfg_path.write_text(yaml_content)
        orch = PipelineOrchestrator.from_yaml(cfg_path)
        assert orch.genre == "sci_fi"
        state = orch.run()
        assert len(state.shots) > 0
