"""Integration tests for the full agentic pipeline."""

import json
from pathlib import Path

import pytest

from divine_conductor.agents.narrator import NarratorAgent
from divine_conductor.agents.director import DirectorAgent
from divine_conductor.agents.cinematographer import CinematographerAgent
from divine_conductor.models.production import (
    Character,
    EmotionalTone,
    PalettePreset,
    ProductionConfig,
    ProductionState,
)
from divine_conductor.pipeline.orchestrator import PipelineOrchestrator


_GENESIS_PASSAGE = (
    "In the beginning God created the heavens and the earth. "
    "The earth was without form and void, and darkness was over the face of the deep. "
    "And the Spirit of God was hovering over the face of the waters. "
    'And God said, "Let there be light," and there was light.'
)


def _make_config(**kwargs) -> ProductionConfig:
    defaults = dict(name="Test Production", passage_text=_GENESIS_PASSAGE)
    defaults.update(kwargs)
    return ProductionConfig(**defaults)


# ---------------------------------------------------------------------------
# NarratorAgent
# ---------------------------------------------------------------------------


class TestNarratorAgent:
    def test_produces_scenes(self):
        state = ProductionState(config=_make_config())
        state = NarratorAgent().run(state)
        assert len(state.scenes) > 0

    def test_scenes_have_text(self):
        state = ProductionState(config=_make_config())
        state = NarratorAgent().run(state)
        for scene in state.scenes:
            assert scene.text.strip() != ""

    def test_scenes_indexed_sequentially(self):
        state = ProductionState(config=_make_config())
        state = NarratorAgent().run(state)
        for i, scene in enumerate(state.scenes):
            assert scene.index == i

    def test_metadata_recorded(self):
        state = ProductionState(config=_make_config())
        state = NarratorAgent().run(state)
        assert "narrator_sentence_count" in state.metadata

    def test_character_assigned_to_scene(self):
        char = Character(id="god_voice", name="God", description="Divine presence.")
        config = _make_config(
            passage_text="God spoke and the world listened.",
            characters=[char],
        )
        state = ProductionState(config=config)
        state = NarratorAgent().run(state)
        scenes_with_god = [s for s in state.scenes if "god_voice" in s.characters]
        assert len(scenes_with_god) > 0

    def test_single_sentence_passage(self):
        config = _make_config(passage_text="God created light.")
        state = ProductionState(config=config)
        state = NarratorAgent().run(state)
        assert len(state.scenes) >= 1


# ---------------------------------------------------------------------------
# DirectorAgent
# ---------------------------------------------------------------------------


class TestDirectorAgent:
    def _state_with_scenes(self) -> ProductionState:
        state = ProductionState(config=_make_config())
        return NarratorAgent().run(state)

    def test_all_scenes_annotated(self):
        state = self._state_with_scenes()
        state = DirectorAgent().run(state)
        for scene in state.scenes:
            assert scene.director_notes != ""

    def test_duration_set(self):
        state = self._state_with_scenes()
        state = DirectorAgent().run(state)
        for scene in state.scenes:
            assert scene.duration_seconds > 0

    def test_documentary_style_longer_shots(self):
        doc_config = _make_config(style="documentary")
        state_doc = ProductionState(config=doc_config)
        state_doc = NarratorAgent().run(state_doc)
        state_doc = DirectorAgent().run(state_doc)

        cin_config = _make_config(style="cinematic")
        state_cin = ProductionState(config=cin_config)
        state_cin = NarratorAgent().run(state_cin)
        state_cin = DirectorAgent().run(state_cin)

        assert state_doc.scenes[0].duration_seconds > state_cin.scenes[0].duration_seconds


# ---------------------------------------------------------------------------
# CinematographerAgent
# ---------------------------------------------------------------------------


class TestCinematographerAgent:
    def _state_with_directed_scenes(self) -> ProductionState:
        state = ProductionState(config=_make_config())
        state = NarratorAgent().run(state)
        state = DirectorAgent().run(state)
        return state

    def test_produces_shots(self):
        state = self._state_with_directed_scenes()
        state = CinematographerAgent().run(state)
        assert len(state.shots) == len(state.scenes)

    def test_shots_have_prompts(self):
        state = self._state_with_directed_scenes()
        state = CinematographerAgent().run(state)
        for shot in state.shots:
            assert shot.prompt.strip() != ""

    def test_shots_per_scene_two(self):
        state = self._state_with_directed_scenes()
        n_scenes = len(state.scenes)
        state = CinematographerAgent(shots_per_scene=2).run(state)
        assert len(state.shots) == n_scenes * 2

    def test_invalid_shots_per_scene(self):
        with pytest.raises(ValueError):
            CinematographerAgent(shots_per_scene=0)


# ---------------------------------------------------------------------------
# PipelineOrchestrator (end-to-end)
# ---------------------------------------------------------------------------


class TestPipelineOrchestrator:
    def test_full_pipeline_runs(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        orch = PipelineOrchestrator(config)
        state = orch.run()
        assert len(state.shots) > 0

    def test_output_json_written(self, tmp_path):
        config = _make_config(output_path=str(tmp_path), output_format="json")
        PipelineOrchestrator(config).run()
        assert (tmp_path / "shots.json").exists()

    def test_output_json_valid(self, tmp_path):
        config = _make_config(output_path=str(tmp_path), output_format="json")
        PipelineOrchestrator(config).run()
        data = json.loads((tmp_path / "shots.json").read_text())
        assert "shots" in data
        assert isinstance(data["shots"], list)
        assert len(data["shots"]) > 0

    def test_output_yaml_written(self, tmp_path):
        config = _make_config(output_path=str(tmp_path), output_format="yaml")
        PipelineOrchestrator(config).run()
        assert (tmp_path / "shots.yaml").exists()

    def test_output_txt_written(self, tmp_path):
        config = _make_config(output_path=str(tmp_path), output_format="txt")
        PipelineOrchestrator(config).run()
        assert (tmp_path / "shots.txt").exists()

    def test_state_summary_correct(self, tmp_path):
        config = _make_config(output_path=str(tmp_path))
        state = PipelineOrchestrator(config).run()
        summary = state.summary()
        assert summary["scenes"] > 0
        assert summary["shots"] > 0
        assert summary["total_duration_seconds"] > 0

    def test_from_yaml(self, tmp_path):
        yaml_content = f"""
pipeline:
  name: "Test from YAML"
  style: cinematic
  aspect_ratio: "16:9"
  fps: 24

passage:
  text: |
    In the beginning God created the heavens and the earth.
    And God said let there be light.

characters: []

consistency:
  palette: warm_golden_dawn
  anchor_shots: true
  character_id_strength: 0.8

output:
  format: json
  path: {tmp_path}
"""
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml_content)
        orch = PipelineOrchestrator.from_yaml(cfg_path)
        state = orch.run()
        assert len(state.shots) > 0

    def test_character_anchors_applied(self, tmp_path):
        char = Character(id="moses", name="Moses", description="Elderly man with wooden staff and flowing robes")
        config = _make_config(
            output_path=str(tmp_path),
            passage_text="Moses led the people through the desert wilderness.",
            characters=[char],
        )
        state = PipelineOrchestrator(config).run()
        # The character description should appear in at least one shot
        combined = " ".join(sh.prompt for sh in state.shots)
        assert "wooden staff" in combined.lower()


# ---------------------------------------------------------------------------
# Subscription gate integration (via PipelineOrchestrator)
# ---------------------------------------------------------------------------


from divine_conductor.models.user import SubscriptionTier, UserSubscription  # noqa: E402


class TestPipelineOrchestratorSubscriptionGate:
    """Verify that passing a UserSubscription trims shot output correctly."""

    _LONG_PASSAGE = (
        "In the beginning God created the heavens and the earth. "
        "The earth was without form and void, and darkness was over the face of the deep. "
        "And the Spirit of God was hovering over the face of the waters. "
        "And God said, let there be light, and there was light. "
        "And God saw that the light was good. "
        "And God separated the light from the darkness and called them day and night."
    )

    def _run_with_tier(self, tmp_path, tier: SubscriptionTier) -> "ProductionState":
        config = _make_config(
            output_path=str(tmp_path),
            passage_text=self._LONG_PASSAGE,
        )
        sub = UserSubscription(user_id="test_user", tier=tier)
        orch = PipelineOrchestrator(config, subscription=sub)
        return orch.run()

    def test_free_tier_caps_duration(self, tmp_path):
        state = self._run_with_tier(tmp_path, SubscriptionTier.FREE)
        total = sum(sh.duration_seconds for sh in state.shots)
        assert total <= 10.0

    def test_ultra_tier_keeps_all_shots(self, tmp_path):
        # Run without gate first to get the full shot count
        config_full = _make_config(
            output_path=str(tmp_path / "full"),
            passage_text=self._LONG_PASSAGE,
        )
        state_full = PipelineOrchestrator(config_full).run()

        state_ultra = self._run_with_tier(tmp_path / "ultra", SubscriptionTier.ULTRA)
        assert len(state_ultra.shots) == len(state_full.shots)

    def test_no_subscription_keeps_all_shots(self, tmp_path):
        """When no subscription is provided the gate should not trim anything."""
        config = _make_config(
            output_path=str(tmp_path),
            passage_text=self._LONG_PASSAGE,
        )
        state = PipelineOrchestrator(config).run()
        assert len(state.shots) > 0
        # Should NOT be trimmed to 10s
        total = sum(sh.duration_seconds for sh in state.shots)
        assert total > 10.0
