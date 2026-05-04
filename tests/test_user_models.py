"""Tests for UserSubscription model and SubscriptionTier enum."""

from __future__ import annotations

import pytest

from divine_conductor.models.user import (
    TIER_MAX_DURATION,
    SubscriptionTier,
    UserSubscription,
)


class TestSubscriptionTier:
    def test_tier_values(self):
        assert SubscriptionTier.FREE.value == "free"
        assert SubscriptionTier.PRO.value == "pro"
        assert SubscriptionTier.ULTRA.value == "ultra"

    def test_all_tiers_in_duration_map(self):
        for tier in SubscriptionTier:
            assert tier in TIER_MAX_DURATION

    def test_free_cap_is_10s(self):
        assert TIER_MAX_DURATION[SubscriptionTier.FREE] == 10.0

    def test_pro_cap_is_300s(self):
        assert TIER_MAX_DURATION[SubscriptionTier.PRO] == 300.0

    def test_ultra_cap_is_none(self):
        assert TIER_MAX_DURATION[SubscriptionTier.ULTRA] is None


class TestUserSubscription:
    def test_defaults_to_free_tier(self):
        sub = UserSubscription(user_id="u1")
        assert sub.tier == SubscriptionTier.FREE
        assert sub.is_active is True
        assert sub.stripe_customer_id is None
        assert sub.stripe_subscription_id is None

    def test_empty_user_id_raises(self):
        with pytest.raises(ValueError, match="user_id"):
            UserSubscription(user_id="")

    def test_max_duration_seconds_free(self):
        sub = UserSubscription(user_id="u1", tier=SubscriptionTier.FREE)
        assert sub.max_duration_seconds == 10.0

    def test_max_duration_seconds_pro(self):
        sub = UserSubscription(user_id="u1", tier=SubscriptionTier.PRO)
        assert sub.max_duration_seconds == 300.0

    def test_max_duration_seconds_ultra(self):
        sub = UserSubscription(user_id="u1", tier=SubscriptionTier.ULTRA)
        assert sub.max_duration_seconds is None

    def test_is_duration_allowed_free_within(self):
        sub = UserSubscription(user_id="u1", tier=SubscriptionTier.FREE)
        assert sub.is_duration_allowed(9.0) is True

    def test_is_duration_allowed_free_exact(self):
        sub = UserSubscription(user_id="u1", tier=SubscriptionTier.FREE)
        assert sub.is_duration_allowed(10.0) is True

    def test_is_duration_allowed_free_over(self):
        sub = UserSubscription(user_id="u1", tier=SubscriptionTier.FREE)
        assert sub.is_duration_allowed(10.1) is False

    def test_is_duration_allowed_ultra_unlimited(self):
        sub = UserSubscription(user_id="u1", tier=SubscriptionTier.ULTRA)
        assert sub.is_duration_allowed(99999.0) is True

    def test_inactive_subscription_falls_back_to_free_cap(self):
        sub = UserSubscription(
            user_id="u1", tier=SubscriptionTier.PRO, is_active=False
        )
        # PRO cap is 300s but since inactive should enforce FREE cap (10s)
        assert sub.is_duration_allowed(11.0) is False
        assert sub.is_duration_allowed(9.0) is True

    def test_to_dict_contains_expected_keys(self):
        sub = UserSubscription(
            user_id="u1",
            tier=SubscriptionTier.PRO,
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
        )
        d = sub.to_dict()
        assert d["user_id"] == "u1"
        assert d["tier"] == "pro"
        assert d["max_duration_seconds"] == 300.0
        assert d["stripe_customer_id"] == "cus_abc"
        assert d["stripe_subscription_id"] == "sub_xyz"
        assert d["is_active"] is True
