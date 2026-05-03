"""Tests for production data models."""

import pytest

from divine_conductor.models.production import (
    CameraAngle,
    Character,
    EmotionalTone,
    PalettePreset,
    ProductionConfig,
    ProductionState,
    Scene,
    Shot,
)


# ---------------------------------------------------------------------------
# Character
# ---------------------------------------------------------------------------


class TestCharacter:
    def test_valid_character(self):
        char = Character(id="moses", name="Moses", description="Elderly man with white beard.")
        assert char.id == "moses"
        assert char.role == "supporting"

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="id"):
            Character(id="", name="Moses", description="desc")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            Character(id="moses", name="Moses", description="")

    def test_description_auto_built_from_canonical_features(self):
        char = Character(
            id="human_male_01",
            name="Adam",
            canonical_features={
                "ethnicity": "undetermined_ancient_mesopotamian_profile",
                "hair": "thick, dark-espresso, shoulder-length, natural waves",
                "eyes": "deep amber, reflective, soulful",
                "build": "athletic, lean-muscular, organic posture",
                "distinguishing_marks": "pristine skin, no scars, natural texture",
            },
            wardrobe_logic={"initial_state": "none"},
        )
        assert char.description != ""
        assert "mesopotamian" in char.description.lower()
        assert "dark-espresso" in char.description
        assert "amber" in char.description
        assert "athletic" in char.description
        assert "pristine skin" in char.description
        assert "wardrobe: none" in char.description

    def test_explicit_description_takes_precedence_over_canonical_features(self):
        char = Character(
            id="human_male_01",
            name="Adam",
            description="Custom explicit description.",
            canonical_features={"hair": "dark"},
        )
        assert char.description == "Custom explicit description."

    def test_missing_description_and_canonical_features_raises(self):
        with pytest.raises(ValueError, match="description"):
            Character(id="x", name="Unknown")

    def test_canonical_features_and_wardrobe_logic_stored(self):
        char = Character(
            id="human_male_01",
            name="Adam",
            canonical_features={"build": "lean"},
            wardrobe_logic={"initial_state": "none", "interaction_rules": {"in_water": "glistens"}},
        )
        assert char.canonical_features["build"] == "lean"
        assert char.wardrobe_logic["interaction_rules"]["in_water"] == "glistens"


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class TestScene:
    def test_valid_scene(self):
        scene = Scene(index=0, text="In the beginning God created the heavens.")
        assert scene.index == 0
        assert scene.tone == EmotionalTone.REVERENT
        assert scene.camera_angle == CameraAngle.WIDE

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="text"):
            Scene(index=0, text="")

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError, match="duration"):
            Scene(index=0, text="Some text.", duration_seconds=-1)

    def test_unique_ids(self):
        s1 = Scene(index=0, text="Text one.")
        s2 = Scene(index=1, text="Text two.")
        assert s1.id != s2.id


# ---------------------------------------------------------------------------
# Shot
# ---------------------------------------------------------------------------


class TestShot:
    def test_valid_shot(self):
        shot = Shot(scene_id="abc", index=0, prompt="Wide shot of the desert.")
        assert shot.scene_id == "abc"
        assert shot.negative_prompt == ""

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="prompt"):
            Shot(scene_id="abc", index=0, prompt="")


# ---------------------------------------------------------------------------
# ProductionConfig
# ---------------------------------------------------------------------------


class TestProductionConfig:
    def test_valid_config(self):
        cfg = ProductionConfig(name="Test", passage_text="Some passage.")
        assert cfg.style == "cinematic"
        assert cfg.palette == PalettePreset.WARM_GOLDEN_DAWN

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError, match="style"):
            ProductionConfig(name="Test", passage_text="text", style="invalid")

    def test_character_strength_out_of_range(self):
        with pytest.raises(ValueError, match="character_id_strength"):
            ProductionConfig(name="T", passage_text="t", character_id_strength=1.5)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            ProductionConfig(name="", passage_text="text")

    def test_empty_passage_raises(self):
        with pytest.raises(ValueError, match="passage_text"):
            ProductionConfig(name="Test", passage_text="")


# ---------------------------------------------------------------------------
# ProductionState
# ---------------------------------------------------------------------------


class TestProductionState:
    def _make_state(self) -> ProductionState:
        cfg = ProductionConfig(name="Test", passage_text="passage")
        return ProductionState(config=cfg)

    def test_initial_state_is_empty(self):
        state = self._make_state()
        assert state.scenes == []
        assert state.shots == []
        assert state.total_duration_seconds == 0.0

    def test_scenes_for_character(self):
        state = self._make_state()
        state.scenes = [
            Scene(index=0, text="a", characters=["moses"]),
            Scene(index=1, text="b", characters=["aaron"]),
            Scene(index=2, text="c", characters=["moses", "aaron"]),
        ]
        assert len(state.scenes_for_character("moses")) == 2
        assert len(state.scenes_for_character("aaron")) == 2
        assert len(state.scenes_for_character("pharaoh")) == 0

    def test_shots_for_scene(self):
        state = self._make_state()
        state.scenes = [Scene(index=0, text="a")]
        scene_id = state.scenes[0].id
        state.shots = [
            Shot(scene_id=scene_id, index=0, prompt="prompt 1"),
            Shot(scene_id="other", index=0, prompt="prompt 2"),
        ]
        assert len(state.shots_for_scene(scene_id)) == 1

    def test_total_duration(self):
        state = self._make_state()
        state.shots = [
            Shot(scene_id="s", index=0, prompt="p", duration_seconds=3.0),
            Shot(scene_id="s", index=1, prompt="p", duration_seconds=4.5),
        ]
        assert state.total_duration_seconds == pytest.approx(7.5)

    def test_summary_keys(self):
        state = self._make_state()
        summary = state.summary()
        for key in ("production", "style", "scenes", "shots", "total_duration_seconds", "consistency_issues"):
            assert key in summary
