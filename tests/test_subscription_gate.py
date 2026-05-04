"""Tests for SubscriptionGate."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from divine_conductor.models.user import SubscriptionTier, UserSubscription
from divine_conductor.pipeline.subscription_gate import (
    SubscriptionGate,
    SubscriptionGateError,
)


# ---------------------------------------------------------------------------
# Minimal shot stub (avoids importing the full production model)
# ---------------------------------------------------------------------------


@dataclass
class _Shot:
    """Lightweight stand-in for production.Shot used in gate tests."""

    duration_seconds: float
    id: str = field(default="shot-stub")


def _shots(*durations: float) -> list[_Shot]:
    return [_Shot(duration_seconds=d, id=f"shot-{i}") for i, d in enumerate(durations)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSubscriptionGate:
    def _gate(self) -> SubscriptionGate:
        return SubscriptionGate()

    # -- store management ------------------------------------------------

    def test_unknown_user_defaults_to_free(self):
        gate = self._gate()
        sub = gate.get_subscription("new_user")
        assert sub.tier == SubscriptionTier.FREE
        assert sub.is_active is True

    def test_upsert_and_retrieve(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="alice", tier=SubscriptionTier.PRO))
        assert gate.get_subscription("alice").tier == SubscriptionTier.PRO

    def test_update_tier_from_stripe(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="bob", tier=SubscriptionTier.FREE))
        sub = gate.update_tier_from_stripe(
            "bob",
            tier=SubscriptionTier.ULTRA,
            stripe_customer_id="cus_001",
            stripe_subscription_id="sub_001",
        )
        assert sub.tier == SubscriptionTier.ULTRA
        assert sub.stripe_customer_id == "cus_001"
        assert sub.is_active is True

    def test_deactivate_sets_inactive(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="carol", tier=SubscriptionTier.PRO))
        gate.deactivate("carol")
        assert gate.get_subscription("carol").is_active is False

    # -- assert_duration_allowed -----------------------------------------

    def test_assert_passes_within_free_cap(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="u1", tier=SubscriptionTier.FREE))
        gate.assert_duration_allowed("u1", 9.0)  # should not raise

    def test_assert_raises_over_free_cap(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="u1", tier=SubscriptionTier.FREE))
        with pytest.raises(SubscriptionGateError, match="cap is 10.0s"):
            gate.assert_duration_allowed("u1", 15.0)

    def test_assert_passes_within_pro_cap(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="u2", tier=SubscriptionTier.PRO))
        gate.assert_duration_allowed("u2", 299.0)

    def test_assert_raises_over_pro_cap(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="u2", tier=SubscriptionTier.PRO))
        with pytest.raises(SubscriptionGateError):
            gate.assert_duration_allowed("u2", 301.0)

    def test_assert_passes_ultra_any_duration(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="u3", tier=SubscriptionTier.ULTRA))
        gate.assert_duration_allowed("u3", 99999.0)

    def test_inactive_pro_enforces_free_cap(self):
        gate = self._gate()
        gate.upsert(
            UserSubscription(user_id="u4", tier=SubscriptionTier.PRO, is_active=False)
        )
        with pytest.raises(SubscriptionGateError):
            gate.assert_duration_allowed("u4", 15.0)

    # -- apply_duration_cap ----------------------------------------------

    def test_cap_trims_excess_shots_free(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="u5", tier=SubscriptionTier.FREE))
        shots = _shots(5.0, 5.0, 5.0)  # total 15s, cap is 10s
        result = gate.apply_duration_cap("u5", shots)
        assert len(result) == 2
        assert sum(s.duration_seconds for s in result) == 10.0

    def test_cap_keeps_all_shots_when_within_limit(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="u6", tier=SubscriptionTier.PRO))
        shots = _shots(5.0, 5.0, 5.0)  # total 15s, cap is 300s
        result = gate.apply_duration_cap("u6", shots)
        assert len(result) == 3

    def test_cap_keeps_all_shots_ultra(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="u7", tier=SubscriptionTier.ULTRA))
        shots = _shots(*[10.0] * 100)
        result = gate.apply_duration_cap("u7", shots)
        assert len(result) == 100

    def test_cap_empty_shot_list(self):
        gate = self._gate()
        gate.upsert(UserSubscription(user_id="u8", tier=SubscriptionTier.FREE))
        assert gate.apply_duration_cap("u8", []) == []

    def test_cap_inactive_enforces_free_cap(self):
        gate = self._gate()
        gate.upsert(
            UserSubscription(user_id="u9", tier=SubscriptionTier.PRO, is_active=False)
        )
        shots = _shots(5.0, 5.0, 5.0)  # total 15s; inactive → FREE cap 10s
        result = gate.apply_duration_cap("u9", shots)
        assert len(result) == 2
