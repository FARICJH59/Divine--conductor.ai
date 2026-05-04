"""Subscription gate for the Divine Conductor pipeline.

The :class:`SubscriptionGate` is the single enforcement point for tier-based
access control.  It sits between the API layer and the ``PipelineOrchestrator``,
answering two questions:

1. Is this user allowed to generate *at all* right now?
2. How much output duration are they allowed to produce?

In production, wire the gate to Supabase/PostgreSQL by replacing the
``_store`` in-memory dict with real database reads.  The interface is
intentionally thin so the swap is trivial.
"""

from __future__ import annotations

import logging

from divine_conductor.models.user import (
    SubscriptionTier,
    TIER_MAX_DURATION,
    UserSubscription,
)

logger = logging.getLogger(__name__)


class SubscriptionGateError(Exception):
    """Raised when a user's subscription does not permit the requested action."""


class SubscriptionGate:
    """Enforces tier limits and manages the in-memory subscription store.

    Args:
        store: Optional pre-populated ``{user_id: UserSubscription}`` mapping.
            Omit to start with an empty store (useful in tests).

    Usage::

        gate = SubscriptionGate()
        gate.upsert(UserSubscription(user_id="alice", tier=SubscriptionTier.PRO))

        # Before running the pipeline:
        sub = gate.get_subscription("alice")
        gate.assert_duration_allowed("alice", requested_seconds=120.0)

        # After generation, trim the shot list to the tier cap:
        allowed = gate.apply_duration_cap("alice", shots)
    """

    def __init__(
        self,
        store: dict[str, UserSubscription] | None = None,
    ) -> None:
        # In production replace this dict with DB look-ups.
        self._store: dict[str, UserSubscription] = store or {}

    # ------------------------------------------------------------------
    # Store management
    # ------------------------------------------------------------------

    def upsert(self, subscription: UserSubscription) -> None:
        """Insert or update a subscription record."""
        self._store[subscription.user_id] = subscription
        logger.debug(
            "Upserted subscription for user=%s tier=%s",
            subscription.user_id,
            subscription.tier.value,
        )

    def get_subscription(self, user_id: str) -> UserSubscription:
        """Return the subscription for *user_id*, defaulting to FREE.

        If the user is not in the store a FREE-tier record is auto-created
        and stored so subsequent calls are consistent.
        """
        if user_id not in self._store:
            logger.info(
                "No subscription found for user=%s; defaulting to FREE.", user_id
            )
            sub = UserSubscription(user_id=user_id, tier=SubscriptionTier.FREE)
            self._store[user_id] = sub
        return self._store[user_id]

    def update_tier_from_stripe(
        self,
        user_id: str,
        *,
        tier: SubscriptionTier,
        stripe_customer_id: str,
        stripe_subscription_id: str,
        is_active: bool = True,
    ) -> UserSubscription:
        """Update the tier for *user_id* after a successful Stripe event.

        This is called by the Stripe webhook handler when
        ``customer.subscription.created`` or ``customer.subscription.updated``
        events are received.

        Returns:
            The updated :class:`UserSubscription`.
        """
        sub = self.get_subscription(user_id)
        sub.tier = tier
        sub.stripe_customer_id = stripe_customer_id
        sub.stripe_subscription_id = stripe_subscription_id
        sub.is_active = is_active
        self._store[user_id] = sub
        logger.info(
            "Subscription updated for user=%s → tier=%s active=%s",
            user_id,
            tier.value,
            is_active,
        )
        return sub

    def deactivate(self, user_id: str) -> None:
        """Mark a subscription as inactive (e.g. after cancellation).

        The user's tier record is kept but ``is_active`` is set to ``False``,
        which causes :meth:`assert_duration_allowed` to enforce FREE limits.
        """
        sub = self.get_subscription(user_id)
        sub.is_active = False
        logger.info("Subscription deactivated for user=%s", user_id)

    # ------------------------------------------------------------------
    # Gate checks
    # ------------------------------------------------------------------

    def assert_duration_allowed(
        self, user_id: str, requested_seconds: float
    ) -> None:
        """Raise :class:`SubscriptionGateError` if the request exceeds the tier cap.

        Args:
            user_id: The user making the generation request.
            requested_seconds: Total duration they are trying to generate.

        Raises:
            SubscriptionGateError: If the requested duration exceeds the tier's
                ``max_duration_seconds``.
        """
        sub = self.get_subscription(user_id)
        if not sub.is_duration_allowed(requested_seconds):
            cap = TIER_MAX_DURATION[sub.tier if sub.is_active else SubscriptionTier.FREE]
            raise SubscriptionGateError(
                f"User '{user_id}' on tier '{sub.tier.value}' "
                f"requested {requested_seconds:.1f}s but the cap is {cap:.1f}s. "
                f"Upgrade at divine-conductor.ai/pricing."
            )

    def apply_duration_cap(self, user_id: str, shots: list) -> list:
        """Return a trimmed shot list that fits within the user's tier cap.

        Unlike :meth:`assert_duration_allowed`, this method silently trims
        instead of raising — useful for the UI path where a graceful truncation
        is preferable to a hard error.

        Args:
            user_id: The user making the generation request.
            shots: Full list of :class:`~divine_conductor.models.production.Shot`
                objects produced by the pipeline.

        Returns:
            A (possibly shorter) list containing only the shots that fit
            within the duration cap.
        """
        sub = self.get_subscription(user_id)
        cap = sub.max_duration_seconds if sub.is_active else TIER_MAX_DURATION[SubscriptionTier.FREE]
        if cap is None:
            return shots  # ULTRA — no cap

        cumulative = 0.0
        allowed: list = []
        for shot in shots:
            duration = getattr(shot, "duration_seconds", 0.0)
            if cumulative + duration > cap:
                logger.info(
                    "Duration cap reached for user=%s (cap=%.1fs). "
                    "Trimming %d trailing shot(s).",
                    user_id,
                    cap,
                    len(shots) - len(allowed),
                )
                break
            allowed.append(shot)
            cumulative += duration
        return allowed
