"""Pipeline orchestrator — wires agents and the consistency engine together."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml  # PyYAML

from divine_conductor.agents.base import BaseAgent
from divine_conductor.agents.cinematographer import CinematographerAgent
from divine_conductor.agents.conflict_resolver import ConflictResolverAgent
from divine_conductor.agents.director import DirectorAgent
from divine_conductor.agents.narrator import NarratorAgent
from divine_conductor.agents.validator import ValidatorAgent
from divine_conductor.consistency.veo_consistency import Veo3ConsistencyEngine
from divine_conductor.models.production import (
    Character,
    PalettePreset,
    PhysicsOverride,
    ProductionConfig,
    ProductionState,
    StyleConflictMetadata,
)
from divine_conductor.pipeline.failure_log import FailureLog

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Drives the full Divine Conductor production pipeline.

    The orchestrator:
    1. Builds the ``ProductionState`` from a ``ProductionConfig``.
    2. Runs ``NarratorAgent`` → ``DirectorAgent`` → ``CinematographerAgent``
       in sequence.
    3. Applies the ``Veo3ConsistencyEngine`` to all generated shots.
    4. Runs a continuity check and stores any issues in the state.
    5. Serialises the result to the configured output path.

    Usage::

        config = ProductionConfig(name="Genesis 1", passage_text="...")
        orchestrator = PipelineOrchestrator(config)
        state = orchestrator.run()

    Or from a YAML config file::

        orchestrator = PipelineOrchestrator.from_yaml("config/example.yaml")
        state = orchestrator.run()
    """

    def __init__(
        self,
        config: ProductionConfig,
        extra_agents: list[BaseAgent] | None = None,
    ) -> None:
        self._config = config
        self._extra_agents = extra_agents or []

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineOrchestrator":
        """Build an orchestrator from a YAML configuration file.

        Args:
            path: Filesystem path to the YAML file.

        Returns:
            A configured ``PipelineOrchestrator`` instance.
        """
        raw = Path(path).read_text(encoding="utf-8")
        data: dict[str, Any] = yaml.safe_load(raw)

        pipeline_cfg = data.get("pipeline", {})
        passage_cfg = data.get("passage", {})
        consistency_cfg = data.get("consistency", {})
        output_cfg = data.get("output", {})

        # Parse characters
        characters: list[Character] = []
        for char_data in data.get("characters", []):
            characters.append(
                Character(
                    id=char_data["id"],
                    name=char_data.get("name", char_data["id"]),
                    description=char_data["description"],
                    role=char_data.get("role", "supporting"),
                )
            )

        # Parse palette preset
        palette_str = consistency_cfg.get("palette", "warm_golden_dawn")
        try:
            palette = PalettePreset(palette_str)
        except ValueError:
            logger.warning(
                "Unknown palette '%s'; falling back to warm_golden_dawn.", palette_str
            )
            palette = PalettePreset.WARM_GOLDEN_DAWN

        # Parse optional style conflict metadata
        style_conflict_cfg = pipeline_cfg.get("style_conflict", {})
        style_conflict: StyleConflictMetadata | None = None
        if style_conflict_cfg:
            physics_cfg = style_conflict_cfg.get("physics_override", {})
            style_conflict = StyleConflictMetadata(
                kinetic_level=str(style_conflict_cfg.get("kinetic_level", "")),
                shutter=str(style_conflict_cfg.get("shutter", "")),
                physics_override=PhysicsOverride(
                    fluid_turbulence=str(physics_cfg.get("fluid_turbulence", "")),
                    debris_density=str(physics_cfg.get("debris_density", "")),
                    gravity_variance=str(physics_cfg.get("gravity_variance", "")),
                ),
            )

        config = ProductionConfig(
            name=pipeline_cfg.get("name", "Untitled Production"),
            passage_text=passage_cfg.get("text", ""),
            style=pipeline_cfg.get("style", "cinematic"),
            genre=pipeline_cfg.get("genre", ""),
            aspect_ratio=pipeline_cfg.get("aspect_ratio", "16:9"),
            fps=int(pipeline_cfg.get("fps", 24)),
            palette=palette,
            character_id_strength=float(
                consistency_cfg.get("character_id_strength", 0.85)
            ),
            anchor_shots=bool(consistency_cfg.get("anchor_shots", True)),
            output_format=output_cfg.get("format", "json"),
            output_path=output_cfg.get("path", "output"),
            characters=characters,
            style_conflict=style_conflict,
        )

        return cls(config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> ProductionState:
        """Execute the full pipeline and return the final ``ProductionState``."""
        logger.info("🎬 Divine Conductor AI — starting pipeline: %s", self._config.name)

        state = ProductionState(config=self._config)

        # ---- Load prior failure log (used by ValidatorAgent suggestions) ----
        failure_log_path = Path(self._config.output_path) / "failure_log.json"
        failure_log = FailureLog.load(failure_log_path)

        # ---- Agent chain ----
        agents: list[BaseAgent] = [
            NarratorAgent(),
            DirectorAgent(),
            CinematographerAgent(),
            ConflictResolverAgent(),
            *self._extra_agents,
        ]
        for agent in agents:
            state = agent(state)

        # ---- Consistency engine ----
        engine = Veo3ConsistencyEngine(
            palette=self._config.palette,
            character_id_strength=self._config.character_id_strength,
        )
        for character in self._config.characters:
            engine.register_character(character)

        state.shots = engine.apply(state.shots)

        issues = engine.check_continuity(state.shots)
        state.consistency_report = [issue.to_dict() for issue in issues]
        if issues:
            logger.warning("%d continuity issue(s) detected.", len(issues))
        else:
            logger.info("No continuity issues detected. ✅")

        # ---- Semantic validation (post-consistency) ----
        validator = ValidatorAgent(failure_log=failure_log)
        state = validator(state)

        # ---- Persist failure log ----
        if len(failure_log) > 0:
            failure_log.save(failure_log_path)
            logger.info(
                "Failure log written to %s (%d record(s)).",
                failure_log_path,
                len(failure_log),
            )

        # ---- Output ----
        self._write_output(state)

        logger.info("🎉 Pipeline complete. Summary: %s", state.summary())
        return state

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _write_output(self, state: ProductionState) -> None:
        """Serialise *state* to the configured output directory."""
        out_dir = Path(self._config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = self._state_to_dict(state)
        fmt = self._config.output_format.lower()

        if fmt == "json":
            out_path = out_dir / "shots.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        elif fmt == "yaml":
            out_path = out_dir / "shots.yaml"
            out_path.write_text(yaml.dump(payload, allow_unicode=True), encoding="utf-8")
        else:
            # Plain text fallback
            out_path = out_dir / "shots.txt"
            lines: list[str] = []
            for shot in payload["shots"]:
                lines.append(f"--- Shot {shot['index']} (scene {shot['scene_id'][:8]}) ---")
                lines.append(shot["prompt"])
                lines.append("")
            out_path.write_text(os.linesep.join(lines), encoding="utf-8")

        logger.info("Output written to %s", out_path)

    @staticmethod
    def _state_to_dict(state: ProductionState) -> dict[str, Any]:
        """Convert *state* to a JSON-serialisable dict."""
        return {
            "production": state.config.name,
            "style": state.config.style,
            "palette": state.config.palette.value,
            "summary": state.summary(),
            "consistency_report": state.consistency_report,
            "shots": [
                {
                    "id": sh.id,
                    "scene_id": sh.scene_id,
                    "index": sh.index,
                    "prompt": sh.prompt,
                    "negative_prompt": sh.negative_prompt,
                    "duration_seconds": sh.duration_seconds,
                    "camera_angle": sh.camera_angle.value,
                }
                for sh in state.shots
            ],
        }
