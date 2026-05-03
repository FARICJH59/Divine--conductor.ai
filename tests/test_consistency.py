"""Tests for the Veo 3.1 consistency engine."""

import pytest

from divine_conductor.consistency.veo_consistency import (
    CharacterAnchor,
    ContinuityIssue,
    Veo3ConsistencyEngine,
)
from divine_conductor.models.production import (
    CameraAngle,
    Character,
    PalettePreset,
    Shot,
)


def _make_shot(prompt: str, scene_id: str = "s1", index: int = 0) -> Shot:
    return Shot(scene_id=scene_id, index=index, prompt=prompt)


class TestCharacterAnchor:
    def test_valid_anchor(self):
        anchor = CharacterAnchor(character_id="moses", anchor_text="Old man.", strength=0.8)
        assert anchor.strength == 0.8

    def test_invalid_strength_raises(self):
        with pytest.raises(ValueError, match="strength"):
            CharacterAnchor(character_id="x", anchor_text="desc", strength=1.5)


class TestVeo3ConsistencyEngineInit:
    def test_default_palette(self):
        engine = Veo3ConsistencyEngine()
        assert engine.palette_anchor.preset == PalettePreset.WARM_GOLDEN_DAWN

    def test_custom_palette(self):
        engine = Veo3ConsistencyEngine(palette=PalettePreset.MOONLIT_BLUE)
        assert engine.palette_anchor.preset == PalettePreset.MOONLIT_BLUE

    def test_no_character_anchors_initially(self):
        engine = Veo3ConsistencyEngine()
        assert engine.character_anchors == {}


class TestRegisterCharacter:
    def test_register_single_character(self):
        engine = Veo3ConsistencyEngine(character_id_strength=0.9)
        char = Character(id="moses", name="Moses", description="Elderly man with staff.")
        engine.register_character(char)
        assert "moses" in engine.character_anchors
        assert engine.character_anchors["moses"].strength == pytest.approx(0.9)

    def test_register_overwrites_on_duplicate(self):
        engine = Veo3ConsistencyEngine()
        char = Character(id="moses", name="Moses", description="v1")
        char2 = Character(id="moses", name="Moses", description="v2")
        engine.register_character(char)
        engine.register_character(char2)
        assert engine.character_anchors["moses"].anchor_text == "v2"


class TestApply:
    def test_palette_injected_into_prompt(self):
        engine = Veo3ConsistencyEngine(palette=PalettePreset.WARM_GOLDEN_DAWN)
        shot = _make_shot("A vast desert scene at dawn.")
        enriched = engine.apply([shot])
        assert len(enriched) == 1
        assert "warm golden" in enriched[0].prompt.lower() or "amber" in enriched[0].prompt.lower()

    def test_character_anchor_injected_when_id_in_prompt(self):
        engine = Veo3ConsistencyEngine()
        char = Character(id="moses", name="Moses", description="Old bearded man with a wooden staff")
        engine.register_character(char)
        shot = _make_shot("moses stands on the mountain top.")
        enriched = engine.apply([shot])
        assert "wooden staff" in enriched[0].prompt.lower()

    def test_character_anchor_not_injected_when_absent(self):
        engine = Veo3ConsistencyEngine()
        char = Character(id="moses", name="Moses", description="Old bearded man")
        engine.register_character(char)
        shot = _make_shot("Aaron raises his hands to the sky.")
        enriched = engine.apply([shot])
        # 'Old bearded man' should NOT be in the prompt since 'moses' isn't mentioned
        assert "old bearded man" not in enriched[0].prompt.lower()

    def test_consistency_anchors_dict_populated(self):
        engine = Veo3ConsistencyEngine()
        shot = _make_shot("Some scene.")
        enriched = engine.apply([shot])
        assert "palette" in enriched[0].consistency_anchors
        assert enriched[0].consistency_anchors["veo_model"] == "veo-3.1"

    def test_original_shots_not_mutated(self):
        engine = Veo3ConsistencyEngine()
        shot = _make_shot("Original prompt.")
        original_prompt = shot.prompt
        engine.apply([shot])
        assert shot.prompt == original_prompt

    def test_shot_id_preserved(self):
        engine = Veo3ConsistencyEngine()
        shot = _make_shot("A scene.")
        original_id = shot.id
        enriched = engine.apply([shot])
        assert enriched[0].id == original_id


class TestCheckContinuity:
    def test_no_issues_for_consistent_shots(self):
        engine = Veo3ConsistencyEngine()
        shots = [
            _make_shot("A sunrise scene in the desert.", index=0),
            _make_shot("The sun rises higher over the dunes.", index=1),
        ]
        issues = engine.check_continuity(shots)
        assert issues == []

    def test_detects_time_of_day_contradiction(self):
        engine = Veo3ConsistencyEngine()
        shots = [
            _make_shot("The people gather at dawn to worship.", index=0),
            _make_shot("Under the midnight stars the prophet sleeps.", index=1),
        ]
        issues = engine.check_continuity(shots)
        assert any(i.category == "time_of_day" for i in issues)

    def test_detects_weather_contradiction(self):
        engine = Veo3ConsistencyEngine()
        shots = [
            _make_shot("A clear sky over Jerusalem, sunny afternoon.", index=0),
            _make_shot("A violent storm tears through the marketplace.", index=1),
        ]
        issues = engine.check_continuity(shots)
        assert any(i.category == "weather" for i in issues)

    def test_no_issues_for_single_shot(self):
        engine = Veo3ConsistencyEngine()
        shot = _make_shot("A single dawn shot.")
        issues = engine.check_continuity([shot])
        assert issues == []

    def test_continuity_issue_to_dict(self):
        issue = ContinuityIssue(
            shot_a_id="aaa", shot_b_id="bbb", category="time_of_day", detail="Conflict."
        )
        d = issue.to_dict()
        assert d["shot_a_id"] == "aaa"
        assert d["category"] == "time_of_day"


# ---------------------------------------------------------------------------
# Wardrobe interaction rules
# ---------------------------------------------------------------------------


def _make_human_male_01() -> Character:
    return Character(
        id="human_male_01",
        name="Adam",
        canonical_features={
            "ethnicity": "undetermined_ancient_mesopotamian_profile",
            "hair": "thick, dark-espresso, shoulder-length, natural waves",
            "eyes": "deep amber, reflective, soulful",
            "build": "athletic, lean-muscular, organic posture",
            "distinguishing_marks": "pristine skin, no scars, natural texture",
        },
        wardrobe_logic={
            "initial_state": "none",
            "interaction_rules": {
                "in_water": "hair clings to neck, skin glistens with droplets",
                "in_sunlight": "warm subsurface scattering on skin edges",
                "in_shadow": "sharp rim lighting to define muscle anatomy",
            },
        },
    )


class TestWardrobeRules:
    def test_wardrobe_rules_stored_on_anchor(self):
        engine = Veo3ConsistencyEngine()
        char = _make_human_male_01()
        engine.register_character(char)
        anchor = engine.character_anchors["human_male_01"]
        assert anchor.wardrobe_rules["in_water"] == "hair clings to neck, skin glistens with droplets"
        assert anchor.wardrobe_rules["in_sunlight"] == "warm subsurface scattering on skin edges"
        assert anchor.wardrobe_rules["in_shadow"] == "sharp rim lighting to define muscle anatomy"

    def test_water_rule_injected_when_water_in_prompt(self):
        engine = Veo3ConsistencyEngine()
        engine.register_character(_make_human_male_01())
        shot = _make_shot("human_male_01 wades through a river at dawn.")
        enriched = engine.apply([shot])
        assert "glistens with droplets" in enriched[0].prompt

    def test_sunlight_rule_injected_when_sun_in_prompt(self):
        engine = Veo3ConsistencyEngine()
        engine.register_character(_make_human_male_01())
        shot = _make_shot("human_male_01 stands in bright sunlight on the hillside.")
        enriched = engine.apply([shot])
        assert "subsurface scattering" in enriched[0].prompt

    def test_shadow_rule_injected_when_shadow_in_prompt(self):
        engine = Veo3ConsistencyEngine()
        engine.register_character(_make_human_male_01())
        shot = _make_shot("human_male_01 walks through deep shadow between the trees.")
        enriched = engine.apply([shot])
        assert "rim lighting" in enriched[0].prompt

    def test_wardrobe_rule_not_injected_when_no_env_match(self):
        engine = Veo3ConsistencyEngine()
        engine.register_character(_make_human_male_01())
        shot = _make_shot("human_male_01 stands on a plain hilltop.")
        enriched = engine.apply([shot])
        assert "glistens with droplets" not in enriched[0].prompt
        assert "subsurface scattering" not in enriched[0].prompt
        assert "rim lighting" not in enriched[0].prompt

    def test_wardrobe_anchor_key_added_to_consistency_dict(self):
        engine = Veo3ConsistencyEngine()
        engine.register_character(_make_human_male_01())
        shot = _make_shot("human_male_01 rests in the shade.")
        enriched = engine.apply([shot])
        assert "wardrobe:human_male_01:in_shadow" in enriched[0].consistency_anchors

    def test_character_without_wardrobe_logic_unaffected(self):
        engine = Veo3ConsistencyEngine()
        char = Character(id="moses", name="Moses", description="Elderly bearded man.")
        engine.register_character(char)
        anchor = engine.character_anchors["moses"]
        assert anchor.wardrobe_rules == {}
