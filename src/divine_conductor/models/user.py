"""User subscription models for the Divine Conductor SaaS tier system.

Defines the canonical subscription tiers and the per-user subscription record.
In production, ``UserSubscription`` objects are persisted in Supabase/PostgreSQL;
in development / unit tests they live in memory via the in-memory store in
:mod:`divine_conductor.pipeline.subscription_gate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------


class SubscriptionTier(str, Enum):
    """Available subscription tiers.

    Each tier maps to a ``max_duration_seconds`` cap enforced by the
    :class:`~divine_conductor.pipeline.subscription_gate.SubscriptionGate`.

    ====== ============ =======================================================
    Tier   Max duration Notes
    ====== ============ =======================================================
    FREE   10 s         Watermarked output; no commercial use.
    PRO    300 s (5 m)  Full resolution; single seat.
    ULTRA  unlimited    Feature-length; multi-seat; priority generation queue.
    ====== ============ =======================================================
    """

    FREE = "free"
    PRO = "pro"
    ULTRA = "ultra"


# Canonical duration cap per tier in seconds (``None`` = unlimited).
TIER_MAX_DURATION: dict[SubscriptionTier, float | None] = {
    SubscriptionTier.FREE: 10.0,
    SubscriptionTier.PRO: 300.0,
    SubscriptionTier.ULTRA: None,
}

# Stripe price IDs — override via environment variables in production.
TIER_STRIPE_PRICE_ID: dict[SubscriptionTier, str | None] = {
    SubscriptionTier.FREE: None,  # no payment required
    SubscriptionTier.PRO: "price_pro_placeholder",
    SubscriptionTier.ULTRA: "price_ultra_placeholder",
}


# ---------------------------------------------------------------------------
# User subscription record
# ---------------------------------------------------------------------------


@dataclass
class UserSubscription:
    """Persisted subscription record for a single user.

    Attributes:
        user_id: Application-level user identifier (e.g. Supabase UUID).
        tier: The user's active subscription tier.
        stripe_customer_id: Stripe Customer object ID (``cus_…``).
            ``None`` for free-tier users who have never checked out.
        stripe_subscription_id: Stripe Subscription object ID (``sub_…``).
            ``None`` when no active paid subscription exists.
        is_active: ``True`` while the subscription is current (not cancelled /
            past-due). Always ``True`` for the free tier.
    """

    user_id: str
    tier: SubscriptionTier = SubscriptionTier.FREE
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None
    is_active: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("UserSubscription.user_id must not be empty.")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def max_duration_seconds(self) -> float | None:
        """Return the maximum allowed production duration for this tier."""
        return TIER_MAX_DURATION[self.tier]

    def is_duration_allowed(self, requested_seconds: float) -> bool:
        """Return ``True`` if *requested_seconds* is within the tier limit.

        A free/pro tier that is not active (e.g. subscription lapsed) falls
        back to the ``FREE`` cap as a safety measure.
        """
        if not self.is_active:
            cap = TIER_MAX_DURATION[SubscriptionTier.FREE]
        else:
            cap = self.max_duration_seconds
        if cap is None:
            return True
        return requested_seconds <= cap

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "user_id": self.user_id,
            "tier": self.tier.value,
            "max_duration_seconds": self.max_duration_seconds,
            "stripe_customer_id": self.stripe_customer_id,
            "stripe_subscription_id": self.stripe_subscription_id,
            "is_active": self.is_active,
        }
