"""FailureLog — persistent store of pipeline validation failures.

Records failures detected by the ValidatorAgent so that the orchestrator
can load prior run data and suggest ``motion_bucket`` adjustments for
the next render pass.

Usage::

    log = FailureLog()
    log.record(FailureRecord(
        shot_id="abc",
        scene_id="xyz",
        failure_type=FailureType.TEMPORAL_CONFLICT,
        detected_value="motion blur present",
        expected_value="freeze-frame sharp",
        motion_bucket_delta=-0.1,
    ))
    log.save("output/failure_log.json")

    # Next run:
    log = FailureLog.load("output/failure_log.json")
    delta = log.suggest_adjustment("abc")  # cumulative delta for that shot
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Failure type enumeration
# ---------------------------------------------------------------------------


class FailureType(str, Enum):
    """Categorises the kind of validation failure recorded by the ValidatorAgent."""

    TEMPORAL_CONFLICT = "TEMPORAL_CONFLICT"
    """Fast shutter / kinetic settings contradicted by motion-blur leakage."""

    HALLUCINATION = "HALLUCINATION"
    """AI produced style-inappropriate content (e.g. VFX fire in a biblical scene)."""

    CHARACTER_ANCHOR_DRIFT = "CHARACTER_ANCHOR_DRIFT"
    """A character's visual identity has merged with or been absorbed by the environment."""


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------


@dataclass
class FailureRecord:
    """A single validated failure captured during a pipeline run.

    Attributes:
        shot_id: ID of the ``Shot`` where the failure was detected.
        scene_id: ID of the parent ``Scene``.
        failure_type: Classification of the failure.
        detected_value: What the validator actually found.
        expected_value: What the validator expected to find.
        motion_bucket_delta: Suggested adjustment to the model's motion bucket
            setting for the next render of this shot.  Negative values slow
            motion; positive values increase it.
        timestamp: ISO-8601 UTC timestamp of when the failure was recorded.
    """

    shot_id: str
    scene_id: str
    failure_type: FailureType
    detected_value: str
    expected_value: str
    motion_bucket_delta: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "shot_id": self.shot_id,
            "scene_id": self.scene_id,
            "failure_type": self.failure_type.value,
            "detected_value": self.detected_value,
            "expected_value": self.expected_value,
            "motion_bucket_delta": self.motion_bucket_delta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FailureRecord":
        return cls(
            timestamp=data["timestamp"],
            shot_id=data["shot_id"],
            scene_id=data["scene_id"],
            failure_type=FailureType(data["failure_type"]),
            detected_value=data["detected_value"],
            expected_value=data["expected_value"],
            motion_bucket_delta=float(data["motion_bucket_delta"]),
        )


# ---------------------------------------------------------------------------
# FailureLog
# ---------------------------------------------------------------------------


class FailureLog:
    """Accumulates ``FailureRecord`` objects and persists them as JSON.

    The log is append-safe: ``load()`` reads existing records and new ones
    are appended via ``record()``.  ``suggest_adjustment()`` aggregates the
    cumulative ``motion_bucket_delta`` across all failures for a given shot,
    giving the orchestrator a single nudge value for the next render pass.
    """

    def __init__(self) -> None:
        self._records: list[FailureRecord] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, entry: FailureRecord) -> None:
        """Append a new failure record to the in-memory log."""
        self._records.append(entry)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Serialise all records to a JSON file at *path*.

        Parent directories are created automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([r.to_dict() for r in self._records], indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "FailureLog":
        """Load records from a JSON file.

        Returns an empty log if the file does not exist.
        """
        path = Path(path)
        log = cls()
        if not path.exists():
            return log
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            log._records.append(FailureRecord.from_dict(item))
        return log

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[FailureRecord]:
        """Return a snapshot of all accumulated records."""
        return list(self._records)

    def suggest_adjustment(self, shot_id: str) -> float:
        """Return the cumulative ``motion_bucket_delta`` for *shot_id*.

        Sums the deltas from all past failures on the same shot so that
        repeated failures compound the correction signal.  Returns ``0.0``
        when no failures have been recorded for the shot.
        """
        return sum(
            r.motion_bucket_delta for r in self._records if r.shot_id == shot_id
        )

    def failures_by_type(self, failure_type: FailureType) -> list[FailureRecord]:
        """Return all records matching *failure_type*."""
        return [r for r in self._records if r.failure_type == failure_type]

    def __len__(self) -> int:
        return len(self._records)
