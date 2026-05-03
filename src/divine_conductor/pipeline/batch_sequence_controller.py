"""BatchSequenceController — divides long productions into sequential ~10-second blocks.

This module implements two core concepts from the Divine Conductor multi-block
architecture:

* **Production Blocks** — approximately 10-second segments of the overall
  passage, each processed as an independent pipeline run.
* **Recursive Continuity** — the final shot of block N provides a visual seed
  (``ContinuitySeed``) that is injected into block N+1's first shot prompt,
  preventing "jump-cut" resets at block boundaries.
* **Manifest** — a JSON-serialisable record of every block's shots and metadata,
  enabling downstream stitching tools to reassemble the full production.
* **Hard Reset** — the ``ValidatorAgent`` is called in strict mode every
  :data:`HARD_RESET_INTERVAL_SECONDS` of cumulative footage to prevent
  semantic drift.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from divine_conductor.agents.cinematographer import CinematographerAgent
from divine_conductor.agents.director import DirectorAgent
from divine_conductor.agents.narrator import NarratorAgent
from divine_conductor.agents.validator import ValidatorAgent, HARD_RESET_INTERVAL_SECONDS
from divine_conductor.consistency.veo_consistency import Veo3ConsistencyEngine
from divine_conductor.models.production import (
    Character,
    ProductionConfig,
    ProductionState,
    Shot,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Target duration for each production block (seconds).
BLOCK_DURATION_SECONDS: float = 10.0

# Sentence boundary splitter (reused from NarratorAgent's pattern)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|(?:\d+:\d+\s+)")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ContinuitySeed:
    """Visual context extracted from the last shot of a production block.

    This seed is forwarded to the next block so its first shot can reference
    the closing visual state, ensuring seamless cross-block continuity.

    Attributes:
        block_index: Index of the block that generated this seed.
        setting: Setting text of the final shot's parent scene.
        prompt_tail: Last 120 characters of the final shot's prompt (key
            visual descriptors).
        camera_angle: Camera angle of the final shot.
        palette: Palette anchor description from the consistency engine.
        motion_bucket: Motion bucket value of the final shot.
    """

    block_index: int
    setting: str
    prompt_tail: str
    camera_angle: str
    palette: str
    motion_bucket: int = 127


@dataclass
class ProductionBlock:
    """A single ~10-second segment of the overall production.

    Attributes:
        index: Zero-based block position in the sequence.
        passage_excerpt: The passage text covered by this block.
        shots: Fully-processed shots for this block.
        continuity_seed: Visual seed extracted from the final shot for use
            by the next block.
        continuity_issues: Continuity issues reported by the consistency engine.
        manifest_entry: JSON-serialisable summary record for the manifest.
    """

    index: int
    passage_excerpt: str
    shots: list[Shot] = field(default_factory=list)
    continuity_seed: ContinuitySeed | None = None
    continuity_issues: list[dict[str, Any]] = field(default_factory=list)
    manifest_entry: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class BatchSequenceController:
    """Processes a long passage as a sequence of ~10-second production blocks.

    The controller:

    1. Splits ``config.passage_text`` into groups of sentences that sum to
       approximately :data:`BLOCK_DURATION_SECONDS` of screen time.
    2. For each block, runs the standard agent chain
       (NarratorAgent → DirectorAgent → CinematographerAgent) plus the
       Veo3ConsistencyEngine.
    3. Forwards the :class:`ContinuitySeed` from block N to block N+1
       (Recursive Continuity), prepending a seed-derived context phrase to
       the first shot's prompt.
    4. Calls :meth:`~divine_conductor.agents.validator.ValidatorAgent.hard_reset_check`
       every :data:`HARD_RESET_INTERVAL_SECONDS` of cumulative footage.
    5. Assembles a manifest and (optionally) writes it to disk.

    Args:
        config: Production configuration. ``passage_text`` is the full text
            to be divided.
        extra_agents: Optional additional agents inserted after
            ``CinematographerAgent`` in every block's pipeline.
        block_duration: Target duration per block in seconds (default 10).
    """

    def __init__(
        self,
        config: ProductionConfig,
        extra_agents: list[Any] | None = None,
        block_duration: float = BLOCK_DURATION_SECONDS,
    ) -> None:
        self._config = config
        self._extra_agents = extra_agents or []
        self._block_duration = block_duration
        self._validator = ValidatorAgent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[ProductionBlock]:
        """Execute the full multi-block pipeline.

        Returns:
            An ordered list of :class:`ProductionBlock` objects, one per
            ~10-second segment.
        """
        logger.info(
            "🎬 BatchSequenceController — starting multi-block run for: %s",
            self._config.name,
        )

        excerpts = self._split_passage(self._config.passage_text)
        logger.info("Passage divided into %d block(s).", len(excerpts))

        blocks: list[ProductionBlock] = []
        seed: ContinuitySeed | None = None
        cumulative_seconds: float = 0.0
        hard_reset_bucket: float = 0.0  # tracks seconds since last Hard Reset

        for block_idx, excerpt in enumerate(excerpts):
            logger.info("Processing block %d/%d…", block_idx + 1, len(excerpts))

            block = self._process_block(block_idx, excerpt, seed)
            blocks.append(block)

            # Accumulate durations
            block_duration = sum(s.duration_seconds for s in block.shots)
            cumulative_seconds += block_duration
            hard_reset_bucket += block_duration

            # Carry the continuity seed forward
            seed = block.continuity_seed

            # Hard Reset check every HARD_RESET_INTERVAL_SECONDS
            if hard_reset_bucket >= HARD_RESET_INTERVAL_SECONDS:
                self._trigger_hard_reset(blocks)
                hard_reset_bucket -= HARD_RESET_INTERVAL_SECONDS

        manifest = self._build_manifest(blocks, cumulative_seconds)
        self._write_manifest(manifest)

        logger.info(
            "✅ BatchSequenceController complete — %d blocks, %.1fs total footage.",
            len(blocks),
            cumulative_seconds,
        )
        return blocks

    def all_shots(self, blocks: list[ProductionBlock]) -> list[Shot]:
        """Flatten all shots from *blocks* into a single ordered list."""
        return [shot for block in blocks for shot in block.shots]

    # ------------------------------------------------------------------
    # Private: block processing
    # ------------------------------------------------------------------

    def _process_block(
        self,
        block_idx: int,
        excerpt: str,
        seed: ContinuitySeed | None,
    ) -> ProductionBlock:
        """Run the pipeline for a single block and return a ``ProductionBlock``."""
        # Build a per-block config with the excerpt as the passage text
        block_config = ProductionConfig(
            name=f"{self._config.name} — Block {block_idx + 1}",
            passage_text=excerpt,
            style=self._config.style,
            aspect_ratio=self._config.aspect_ratio,
            fps=self._config.fps,
            palette=self._config.palette,
            character_id_strength=self._config.character_id_strength,
            anchor_shots=self._config.anchor_shots,
            output_format=self._config.output_format,
            output_path=self._config.output_path,
            characters=list(self._config.characters),
        )

        state = ProductionState(config=block_config)

        # Agent chain
        for agent in [NarratorAgent(), DirectorAgent(), CinematographerAgent()]:
            state = agent(state)
        for agent in self._extra_agents:
            state = agent(state)

        # Consistency engine
        engine = Veo3ConsistencyEngine(
            palette=self._config.palette,
            character_id_strength=self._config.character_id_strength,
        )
        for character in self._config.characters:
            engine.register_character(character)
        state.shots = engine.apply(state.shots)

        continuity_issues = [i.to_dict() for i in engine.check_continuity(state.shots)]

        # Inject continuity seed from previous block into first shot
        if seed is not None and state.shots:
            state.shots[0] = self._apply_seed(state.shots[0], seed)

        # Extract seed for next block
        new_seed = self._extract_seed(block_idx, state)

        block = ProductionBlock(
            index=block_idx,
            passage_excerpt=excerpt,
            shots=state.shots,
            continuity_seed=new_seed,
            continuity_issues=continuity_issues,
        )
        block.manifest_entry = self._build_block_manifest_entry(block)
        return block

    # ------------------------------------------------------------------
    # Private: Recursive Continuity
    # ------------------------------------------------------------------

    def _apply_seed(self, shot: Shot, seed: ContinuitySeed) -> Shot:
        """Prepend the seed context to *shot*'s prompt (Recursive Continuity)."""
        seed_phrase = (
            f"[Continuing from block {seed.block_index}: {seed.prompt_tail}, "
            f"{seed.setting}, {seed.camera_angle} camera angle]"
        )
        return Shot(
            scene_id=shot.scene_id,
            index=shot.index,
            prompt=f"{seed_phrase}, {shot.prompt}",
            negative_prompt=shot.negative_prompt,
            duration_seconds=shot.duration_seconds,
            camera_angle=shot.camera_angle,
            motion_bucket=shot.motion_bucket,
            consistency_anchors=dict(shot.consistency_anchors),
            id=shot.id,
        )

    def _extract_seed(
        self, block_idx: int, state: ProductionState
    ) -> ContinuitySeed | None:
        """Extract a :class:`ContinuitySeed` from the last shot in *state*."""
        if not state.shots:
            return None

        last_shot = state.shots[-1]
        last_scene = next(
            (sc for sc in reversed(state.scenes) if sc.id == last_shot.scene_id),
            None,
        )
        setting = last_scene.setting if last_scene else ""
        prompt_tail = last_shot.prompt[-120:].strip()
        palette = last_shot.consistency_anchors.get("palette", "")

        return ContinuitySeed(
            block_index=block_idx,
            setting=setting,
            prompt_tail=prompt_tail,
            camera_angle=last_shot.camera_angle.value,
            palette=palette,
            motion_bucket=last_shot.motion_bucket,
        )

    # ------------------------------------------------------------------
    # Private: Hard Reset
    # ------------------------------------------------------------------

    def _trigger_hard_reset(self, blocks: list[ProductionBlock]) -> None:
        """Run a Hard Reset validation pass over all shots accumulated so far."""
        all_accumulated = [s for b in blocks for s in b.shots]
        logger.info(
            "⚡ Hard Reset triggered — validating %d shots (%.1f s cumulative).",
            len(all_accumulated),
            sum(s.duration_seconds for s in all_accumulated),
        )
        issues = self._validator.hard_reset_check(all_accumulated)
        if issues:
            logger.warning(
                "Hard Reset found %d issue(s). Review validation_issues in state metadata.",
                len(issues),
            )
        else:
            logger.info("Hard Reset: no semantic drift detected. ✅")

    # ------------------------------------------------------------------
    # Private: Manifest
    # ------------------------------------------------------------------

    def _build_manifest(
        self, blocks: list[ProductionBlock], total_seconds: float
    ) -> dict[str, Any]:
        return {
            "production": self._config.name,
            "style": self._config.style,
            "total_duration_seconds": round(total_seconds, 2),
            "block_count": len(blocks),
            "blocks": [b.manifest_entry for b in blocks],
        }

    @staticmethod
    def _build_block_manifest_entry(block: ProductionBlock) -> dict[str, Any]:
        return {
            "block_index": block.index,
            "passage_excerpt": block.passage_excerpt[:120],
            "shot_count": len(block.shots),
            "duration_seconds": round(
                sum(s.duration_seconds for s in block.shots), 2
            ),
            "continuity_issues": len(block.continuity_issues),
            "shot_ids": [s.id for s in block.shots],
        }

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        """Write *manifest* to the configured output directory."""
        out_dir = Path(self._config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "batch_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Batch manifest written to %s", manifest_path)

    # ------------------------------------------------------------------
    # Private: passage splitting
    # ------------------------------------------------------------------

    def _split_passage(self, passage_text: str) -> list[str]:
        """Split *passage_text* into excerpts targeting ``self._block_duration`` each.

        Sentences are grouped until their estimated total duration (at
        ``default_duration_seconds`` per sentence) reaches the block target.
        Estimates are based on 5 s per sentence (the NarratorAgent default).
        """
        sentences = self._sentence_split(passage_text)
        if not sentences:
            return [passage_text]

        # Default DirectorAgent base duration (cinematic = 1×5s, doc = 1.4×5s…)
        _STYLE_PACE: dict[str, float] = {
            "cinematic": 1.0,
            "documentary": 1.4,
            "animated": 0.75,
        }
        seconds_per_sentence = 5.0 * _STYLE_PACE.get(self._config.style, 1.0)
        sentences_per_block = max(1, round(self._block_duration / seconds_per_sentence))

        excerpts: list[str] = []
        for i in range(0, len(sentences), sentences_per_block):
            chunk = sentences[i : i + sentences_per_block]
            excerpts.append(" ".join(chunk))
        return excerpts

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        """Split *text* into individual sentences."""
        raw = _SENTENCE_RE.split(text.strip())
        return [s.strip() for s in raw if s.strip()]
