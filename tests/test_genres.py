"""Tests for GenreFactory, Genre model, and cyber_noir genre integration."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from divine_conductor.consistency.veo_consistency import Veo3ConsistencyEngine
from divine_conductor.core.factory import GenreFactory
from divine_conductor.models.production import (
    Genre,
    PalettePreset,
    ProductionConfig,
    Shot,
)
from divine_conductor.pipeline.orchestrator import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shot(prompt: str, scene_id: str = "s1", index: int = 0) -> Shot:
    return Shot(scene_id=scene_id, index=index, prompt=prompt)


def _cyber_noir_genre() -> Genre:
    return Genre(
        key="cyber_noir",
        visual_anchors=["neon reflections", "digital rain", "anamorphic flares"],
        lighting="high-contrast cyan and crimson, flickering overhead drones",
        camera_tech="Shot on 35mm film, high grain, 1.33:1 aspect ratio",
        wardrobe_modifier="weathered leather trench coat, integrated cyber-link",
    )


# ---------------------------------------------------------------------------
# Genre dataclass
# ---------------------------------------------------------------------------


class TestGenreModel:
    def test_valid_genre(self):
        g = _cyber_noir_genre()
        assert g.key == "cyber_noir"
        assert "neon reflections" in g.visual_anchors
        assert "digital rain" in g.visual_anchors
        assert "anamorphic flares" in g.visual_anchors
        assert "cyan" in g.lighting
        assert "35mm" in g.camera_tech
        assert "trench coat" in g.wardrobe_modifier

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="key"):
            Genre(key="")

    def test_defaults_are_empty(self):
        g = Genre(key="minimal")
        assert g.visual_anchors == []
        assert g.lighting == ""
        assert g.camera_tech == ""
        assert g.wardrobe_modifier == ""


# ---------------------------------------------------------------------------
# GenreFactory
# ---------------------------------------------------------------------------


class TestGenreFactory:
    def test_load_cyber_noir(self, tmp_path):
        """Factory loads cyber_noir.yaml and returns a correct Genre."""
        # Write a real-shaped YAML
        (tmp_path / "cyber_noir.yaml").write_text(
            textwrap.dedent("""\
                key: cyber_noir
                visual_anchors:
                  - neon reflections
                  - digital rain
                  - anamorphic flares
                lighting: "high-contrast cyan and crimson, flickering overhead drones"
                camera_tech: "Shot on 35mm film, high grain, 1.33:1 aspect ratio"
                wardrobe_modifier: "weathered leather trench coat, integrated cyber-link"
            """),
            encoding="utf-8",
        )
        factory = GenreFactory(genres_dir=tmp_path)
        genre = factory.load("cyber_noir")
        assert genre.key == "cyber_noir"
        assert "neon reflections" in genre.visual_anchors
        assert "digital rain" in genre.visual_anchors
        assert "anamorphic flares" in genre.visual_anchors
        assert "cyan" in genre.lighting
        assert "35mm" in genre.camera_tech
        assert "trench coat" in genre.wardrobe_modifier

    def test_load_missing_key_raises_file_not_found(self, tmp_path):
        factory = GenreFactory(genres_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="unknown_genre"):
            factory.load("unknown_genre")

    def test_load_all_returns_all_yaml_files(self, tmp_path):
        for name, key in [("alpha.yaml", "alpha"), ("beta.yaml", "beta")]:
            (tmp_path / name).write_text(f"key: {key}\n", encoding="utf-8")
        factory = GenreFactory(genres_dir=tmp_path)
        all_genres = factory.load_all()
        assert set(all_genres.keys()) == {"alpha", "beta"}

    def test_load_all_empty_dir(self, tmp_path):
        factory = GenreFactory(genres_dir=tmp_path)
        assert factory.load_all() == {}

    def test_load_all_missing_dir(self, tmp_path):
        factory = GenreFactory(genres_dir=tmp_path / "nonexistent")
        assert factory.load_all() == {}

    def test_file_stem_used_as_key_fallback(self, tmp_path):
        """If 'key' field is absent the file stem becomes the key."""
        (tmp_path / "mystery.yaml").write_text(
            "visual_anchors: []\n", encoding="utf-8"
        )
        factory = GenreFactory(genres_dir=tmp_path)
        genre = factory.load("mystery")
        assert genre.key == "mystery"

    def test_loads_real_cyber_noir_yaml(self):
        """The committed config/genres/cyber_noir.yaml loads without errors."""
        factory = GenreFactory()
        genre = factory.load("cyber_noir")
        assert genre.key == "cyber_noir"
        assert len(genre.visual_anchors) == 3
        assert genre.lighting != ""
        assert genre.camera_tech != ""
        assert genre.wardrobe_modifier != ""


# ---------------------------------------------------------------------------
# Veo3ConsistencyEngine + Genre injection
# ---------------------------------------------------------------------------


class TestConsistencyEngineWithGenre:
    def test_visual_anchors_injected_into_prompt(self):
        engine = Veo3ConsistencyEngine(genre=_cyber_noir_genre())
        shot = _make_shot("A detective walks down a rain-soaked alley.")
        enriched = engine.apply([shot])
        prompt = enriched[0].prompt
        assert "neon reflections" in prompt
        assert "digital rain" in prompt
        assert "anamorphic flares" in prompt

    def test_lighting_injected_into_prompt(self):
        engine = Veo3ConsistencyEngine(genre=_cyber_noir_genre())
        shot = _make_shot("The city sleeps under a purple haze.")
        enriched = engine.apply([shot])
        assert "cyan and crimson" in enriched[0].prompt

    def test_wardrobe_modifier_injected_into_prompt(self):
        engine = Veo3ConsistencyEngine(genre=_cyber_noir_genre())
        shot = _make_shot("A figure emerges from the shadows.")
        enriched = engine.apply([shot])
        assert "trench coat" in enriched[0].prompt

    def test_camera_tech_replaces_default_quality_tokens(self):
        engine = Veo3ConsistencyEngine(genre=_cyber_noir_genre())
        shot = _make_shot("A neon-lit street at midnight.")
        enriched = engine.apply([shot])
        prompt = enriched[0].prompt
        assert "35mm film" in prompt
        # Default ARRI ALEXA 35 quality suffix should NOT appear
        assert "ARRI ALEXA 35" not in prompt

    def test_no_genre_uses_default_quality_tokens(self):
        engine = Veo3ConsistencyEngine()
        shot = _make_shot("A peaceful dawn over the valley.")
        enriched = engine.apply([shot])
        assert "ARRI ALEXA 35" in enriched[0].prompt

    def test_genre_anchors_stored_in_consistency_dict(self):
        engine = Veo3ConsistencyEngine(genre=_cyber_noir_genre())
        shot = _make_shot("A scene.")
        enriched = engine.apply([shot])
        anchors = enriched[0].consistency_anchors
        assert anchors["genre:key"] == "cyber_noir"
        assert "neon reflections" in anchors["genre:visual_anchors"]
        assert "cyan" in anchors["genre:lighting"]
        assert "trench coat" in anchors["genre:wardrobe_modifier"]
        assert "35mm" in anchors["genre:camera_tech"]

    def test_genre_property(self):
        genre = _cyber_noir_genre()
        engine = Veo3ConsistencyEngine(genre=genre)
        assert engine.genre is genre

    def test_genre_property_none_by_default(self):
        engine = Veo3ConsistencyEngine()
        assert engine.genre is None

    def test_veo_model_anchor_always_present(self):
        engine = Veo3ConsistencyEngine(genre=_cyber_noir_genre())
        shot = _make_shot("A shot.")
        enriched = engine.apply([shot])
        assert enriched[0].consistency_anchors["veo_model"] == "veo-3.1"


# ---------------------------------------------------------------------------
# PipelineOrchestrator — end-to-end with cyber_noir
# ---------------------------------------------------------------------------

_CYBER_PASSAGE = (
    "Rain hammered the neon-soaked streets of Neo-Babylon. "
    "A detective pulled her collar against the drone-lit sky, "
    "jacking into the city's digital nervous system."
)


class TestOrchestratorWithGenre:
    def test_pipeline_runs_with_genre(self, tmp_path):
        config = ProductionConfig(
            name="Cyber Noir Test",
            passage_text=_CYBER_PASSAGE,
            output_path=str(tmp_path),
            genre=_cyber_noir_genre(),
        )
        state = PipelineOrchestrator(config).run()
        assert len(state.shots) > 0

    def test_genre_visual_anchors_present_in_shots(self, tmp_path):
        config = ProductionConfig(
            name="Cyber Noir Test",
            passage_text=_CYBER_PASSAGE,
            output_path=str(tmp_path),
            genre=_cyber_noir_genre(),
        )
        state = PipelineOrchestrator(config).run()
        combined = " ".join(sh.prompt for sh in state.shots)
        assert "neon reflections" in combined
        assert "digital rain" in combined
        assert "anamorphic flares" in combined

    def test_genre_camera_tech_in_shots(self, tmp_path):
        config = ProductionConfig(
            name="Cyber Noir Test",
            passage_text=_CYBER_PASSAGE,
            output_path=str(tmp_path),
            genre=_cyber_noir_genre(),
        )
        state = PipelineOrchestrator(config).run()
        combined = " ".join(sh.prompt for sh in state.shots)
        assert "35mm film" in combined
        assert "ARRI ALEXA 35" not in combined

    def test_from_yaml_string_genre_key(self, tmp_path):
        yaml_content = textwrap.dedent(f"""\
            pipeline:
              name: "Cyber Noir YAML Test"
              style: cinematic
              aspect_ratio: "16:9"
              fps: 24

            passage:
              text: |
                Rain fell on the chrome towers.
                The detective followed the signal into the dark.

            genre: cyber_noir

            characters: []

            consistency:
              palette: moonlit_blue
              anchor_shots: true
              character_id_strength: 0.8

            output:
              format: json
              path: {tmp_path}
        """)
        cfg_path = tmp_path / "cyber_noir_test.yaml"
        cfg_path.write_text(yaml_content)
        state = PipelineOrchestrator.from_yaml(cfg_path).run()
        combined = " ".join(sh.prompt for sh in state.shots)
        assert "neon reflections" in combined or "digital rain" in combined

    def test_from_yaml_inline_genre_dict(self, tmp_path):
        yaml_content = textwrap.dedent(f"""\
            pipeline:
              name: "Inline Genre Test"
              style: cinematic

            passage:
              text: |
                The city hummed with electric dreams.

            genre:
              key: custom_noir
              visual_anchors:
                - holographic overlays
              lighting: deep violet tones
              camera_tech: "anamorphic 2.39:1"
              wardrobe_modifier: "chrome exo-suit"

            characters: []

            consistency:
              palette: moonlit_blue

            output:
              format: json
              path: {tmp_path}
        """)
        cfg_path = tmp_path / "inline_genre_test.yaml"
        cfg_path.write_text(yaml_content)
        state = PipelineOrchestrator.from_yaml(cfg_path).run()
        combined = " ".join(sh.prompt for sh in state.shots)
        assert "holographic overlays" in combined
        assert "chrome exo-suit" in combined

    def test_from_yaml_missing_genre_key_does_not_crash(self, tmp_path):
        yaml_content = textwrap.dedent(f"""\
            pipeline:
              name: "No Genre Test"
              style: cinematic

            passage:
              text: |
                A quiet morning in the desert.

            genre: nonexistent_genre_xyz

            characters: []

            consistency:
              palette: warm_golden_dawn

            output:
              format: json
              path: {tmp_path}
        """)
        cfg_path = tmp_path / "missing_genre_test.yaml"
        cfg_path.write_text(yaml_content)
        # Should not raise — logs a warning and continues without genre
        state = PipelineOrchestrator.from_yaml(cfg_path).run()
        assert len(state.shots) > 0
