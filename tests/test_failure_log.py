"""Unit tests for the FailureLog module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from divine_conductor.pipeline.failure_log import (
    FailureLog,
    FailureRecord,
    FailureType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    shot_id: str = "shot-abc",
    scene_id: str = "scene-xyz",
    failure_type: FailureType = FailureType.TEMPORAL_CONFLICT,
    detected_value: str = "motion blur present",
    expected_value: str = "freeze-frame sharp",
    motion_bucket_delta: float = -0.1,
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> FailureRecord:
    return FailureRecord(
        shot_id=shot_id,
        scene_id=scene_id,
        failure_type=failure_type,
        detected_value=detected_value,
        expected_value=expected_value,
        motion_bucket_delta=motion_bucket_delta,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# FailureRecord
# ---------------------------------------------------------------------------


class TestFailureRecord:
    def test_to_dict_keys(self):
        record = _make_record()
        d = record.to_dict()
        for key in (
            "timestamp",
            "shot_id",
            "scene_id",
            "failure_type",
            "detected_value",
            "expected_value",
            "motion_bucket_delta",
        ):
            assert key in d

    def test_to_dict_failure_type_is_string(self):
        record = _make_record(failure_type=FailureType.HALLUCINATION)
        assert record.to_dict()["failure_type"] == "HALLUCINATION"

    def test_round_trip(self):
        record = _make_record()
        restored = FailureRecord.from_dict(record.to_dict())
        assert restored.shot_id == record.shot_id
        assert restored.failure_type == record.failure_type
        assert restored.motion_bucket_delta == pytest.approx(record.motion_bucket_delta)

    def test_default_timestamp_is_set(self):
        record = FailureRecord(
            shot_id="s",
            scene_id="sc",
            failure_type=FailureType.CHARACTER_ANCHOR_DRIFT,
            detected_value="absent",
            expected_value="present",
            motion_bucket_delta=0.05,
        )
        assert record.timestamp != ""


# ---------------------------------------------------------------------------
# FailureLog — record & query
# ---------------------------------------------------------------------------


class TestFailureLogRecord:
    def test_empty_log_has_zero_len(self):
        log = FailureLog()
        assert len(log) == 0

    def test_record_appends(self):
        log = FailureLog()
        log.record(_make_record())
        assert len(log) == 1

    def test_records_property_is_snapshot(self):
        log = FailureLog()
        log.record(_make_record())
        snapshot = log.records
        log.record(_make_record(shot_id="other"))
        # Snapshot must not be affected by subsequent additions
        assert len(snapshot) == 1

    def test_failures_by_type(self):
        log = FailureLog()
        log.record(_make_record(failure_type=FailureType.TEMPORAL_CONFLICT))
        log.record(_make_record(failure_type=FailureType.HALLUCINATION))
        log.record(_make_record(failure_type=FailureType.TEMPORAL_CONFLICT))

        assert len(log.failures_by_type(FailureType.TEMPORAL_CONFLICT)) == 2
        assert len(log.failures_by_type(FailureType.HALLUCINATION)) == 1
        assert len(log.failures_by_type(FailureType.CHARACTER_ANCHOR_DRIFT)) == 0


# ---------------------------------------------------------------------------
# FailureLog — suggest_adjustment
# ---------------------------------------------------------------------------


class TestSuggestAdjustment:
    def test_returns_zero_for_unknown_shot(self):
        log = FailureLog()
        assert log.suggest_adjustment("unknown-shot") == pytest.approx(0.0)

    def test_single_record_delta(self):
        log = FailureLog()
        log.record(_make_record(shot_id="s1", motion_bucket_delta=-0.1))
        assert log.suggest_adjustment("s1") == pytest.approx(-0.1)

    def test_multiple_records_same_shot_cumulates(self):
        log = FailureLog()
        log.record(_make_record(shot_id="s1", motion_bucket_delta=-0.1))
        log.record(_make_record(shot_id="s1", motion_bucket_delta=-0.1))
        assert log.suggest_adjustment("s1") == pytest.approx(-0.2)

    def test_different_shots_do_not_interfere(self):
        log = FailureLog()
        log.record(_make_record(shot_id="s1", motion_bucket_delta=-0.1))
        log.record(_make_record(shot_id="s2", motion_bucket_delta=0.05))
        assert log.suggest_adjustment("s1") == pytest.approx(-0.1)
        assert log.suggest_adjustment("s2") == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# FailureLog — persistence
# ---------------------------------------------------------------------------


class TestFailureLogPersistence:
    def test_save_creates_file(self, tmp_path):
        log = FailureLog()
        log.record(_make_record())
        path = tmp_path / "failure_log.json"
        log.save(path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        log = FailureLog()
        log.record(_make_record())
        nested_path = tmp_path / "a" / "b" / "failure_log.json"
        log.save(nested_path)
        assert nested_path.exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        log = FailureLog()
        log.record(_make_record())
        path = tmp_path / "failure_log.json"
        log.save(path)
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_load_non_existent_returns_empty_log(self, tmp_path):
        log = FailureLog.load(tmp_path / "missing.json")
        assert len(log) == 0

    def test_round_trip_persistence(self, tmp_path):
        log = FailureLog()
        log.record(_make_record(shot_id="s1", motion_bucket_delta=-0.1))
        log.record(
            _make_record(
                shot_id="s2",
                failure_type=FailureType.HALLUCINATION,
                motion_bucket_delta=0.0,
            )
        )
        path = tmp_path / "failure_log.json"
        log.save(path)

        loaded = FailureLog.load(path)
        assert len(loaded) == 2
        assert loaded.records[0].shot_id == "s1"
        assert loaded.records[1].failure_type == FailureType.HALLUCINATION

    def test_suggest_adjustment_after_load(self, tmp_path):
        log = FailureLog()
        log.record(_make_record(shot_id="s1", motion_bucket_delta=-0.1))
        path = tmp_path / "failure_log.json"
        log.save(path)

        loaded = FailureLog.load(path)
        assert loaded.suggest_adjustment("s1") == pytest.approx(-0.1)

    def test_all_failure_types_round_trip(self, tmp_path):
        path = tmp_path / "log.json"
        for ft in FailureType:
            log = FailureLog()
            log.record(_make_record(failure_type=ft))
            log.save(path)
            loaded = FailureLog.load(path)
            assert loaded.records[0].failure_type == ft
