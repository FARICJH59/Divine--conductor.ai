"""Divine Conductor — FastAPI subscription backend.

Provides three endpoints that form the Stripe billing loop:

* ``POST /api/checkout``   — Create a Stripe Checkout Session (returns URL).
* ``POST /api/webhook``    — Handle Stripe webhook events (subscription sync).
* ``GET  /api/subscription/{user_id}`` — Inspect a user's current tier.

Environment variables (set in production; stubbed in tests):

.. code-block:: text

    STRIPE_SECRET_KEY       sk_live_… or sk_test_…
    STRIPE_WEBHOOK_SECRET   whsec_…
    STRIPE_SUCCESS_URL      https://your-app.com/success
    STRIPE_CANCEL_URL       https://your-app.com/cancel

Running locally::

    uvicorn api.main:app --reload --port 8000

Or from the repo root after ``pip install -e ".[api]"``::

    divine-conductor-api
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow the package to be imported without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import stripe
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from divine_conductor.models.user import (
    SubscriptionTier,
    TIER_STRIPE_PRICE_ID,
    UserSubscription,
)
from divine_conductor.pipeline.subscription_gate import SubscriptionGate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stripe configuration
# ---------------------------------------------------------------------------

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
_WEBHOOK_SECRET: str = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
_SUCCESS_URL: str = os.environ.get(
    "STRIPE_SUCCESS_URL", "http://localhost:3000/success"
)
_CANCEL_URL: str = os.environ.get(
    "STRIPE_CANCEL_URL", "http://localhost:3000/cancel"
)

# ---------------------------------------------------------------------------
# Shared subscription store
# In production replace with a Supabase/PostgreSQL-backed implementation.
# ---------------------------------------------------------------------------

_gate = SubscriptionGate()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Divine Conductor API",
    description="Subscription management and agentic pipeline gateway.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CheckoutRequest(BaseModel):
    """Body for ``POST /api/checkout``."""

    user_id: str
    tier: SubscriptionTier


class CheckoutResponse(BaseModel):
    """Successful checkout session response."""

    checkout_url: str
    session_id: str


class SubscriptionResponse(BaseModel):
    """Serialised subscription data."""

    user_id: str
    tier: str
    max_duration_seconds: float | None
    stripe_customer_id: str | None
    stripe_subscription_id: str | None
    is_active: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/api/checkout",
    response_model=CheckoutResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a Stripe Checkout Session",
)
async def create_checkout_session(body: CheckoutRequest) -> CheckoutResponse:
    """Create a hosted Stripe Checkout page for the requested tier.

    The caller (Next.js frontend) should redirect the user to the returned
    ``checkout_url``.  After payment, Stripe calls ``POST /api/webhook``
    with a ``customer.subscription.created`` event which updates the DB.

    Args:
        body: ``user_id`` and the target ``tier``.

    Returns:
        The Stripe-hosted checkout URL and session ID.

    Raises:
        HTTPException 400: If the FREE tier is requested (no payment needed).
        HTTPException 502: If Stripe returns an error.
    """
    if body.tier == SubscriptionTier.FREE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The FREE tier requires no payment. No checkout needed.",
        )

    price_id = TIER_STRIPE_PRICE_ID.get(body.tier)
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No Stripe price configured for tier '{body.tier.value}'.",
        )

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{_SUCCESS_URL}?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=_CANCEL_URL,
            metadata={"user_id": body.user_id, "tier": body.tier.value},
        )
    except stripe.StripeError as exc:
        logger.error("Stripe error creating session: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Stripe error: {exc.user_message}",
        ) from exc

    return CheckoutResponse(checkout_url=session.url, session_id=session.id)


@app.post(
    "/api/webhook",
    status_code=status.HTTP_200_OK,
    summary="Stripe webhook receiver",
)
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(alias="stripe-signature", default=""),
) -> JSONResponse:
    """Receive and process Stripe webhook events.

    Stripe sends signed POST requests to this endpoint when subscription
    lifecycle events occur.  The signature is verified against
    ``STRIPE_WEBHOOK_SECRET`` before any processing takes place.

    Handled events:

    * ``customer.subscription.created`` — activate subscription, set tier.
    * ``customer.subscription.updated`` — update tier if plan changed.
    * ``customer.subscription.deleted`` — deactivate subscription.

    All other events are acknowledged with 200 and ignored.

    Raises:
        HTTPException 400: If the Stripe signature is invalid.
    """
    payload = await request.body()

    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, _WEBHOOK_SECRET
        )
    except stripe.SignatureVerificationError as exc:
        logger.warning("Webhook signature verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Stripe signature.",
        ) from exc

    event_type: str = event["type"]
    logger.info("Stripe webhook received: %s", event_type)

    if event_type in (
        "customer.subscription.created",
        "customer.subscription.updated",
    ):
        _handle_subscription_upsert(event["data"]["object"])
    elif event_type == "customer.subscription.deleted":
        _handle_subscription_deleted(event["data"]["object"])

    return JSONResponse({"received": True})


@app.get(
    "/api/subscription/{user_id}",
    response_model=SubscriptionResponse,
    summary="Get user subscription",
)
async def get_subscription(user_id: str) -> SubscriptionResponse:
    """Return the current subscription record for *user_id*.

    If the user has never subscribed, a FREE-tier record is returned.
    """
    sub = _gate.get_subscription(user_id)
    return SubscriptionResponse(
        user_id=sub.user_id,
        tier=sub.tier.value,
        max_duration_seconds=sub.max_duration_seconds,
        stripe_customer_id=sub.stripe_customer_id,
        stripe_subscription_id=sub.stripe_subscription_id,
        is_active=sub.is_active,
    )


# ---------------------------------------------------------------------------
# Internal webhook helpers
# ---------------------------------------------------------------------------


def _handle_subscription_upsert(stripe_sub: dict) -> None:
    """Process ``customer.subscription.created/updated``."""
    user_id: str | None = stripe_sub.get("metadata", {}).get("user_id")
    if not user_id:
        logger.warning("Subscription event missing user_id metadata; skipping.")
        return

    # Map Stripe price ID back to our tier enum.
    items = stripe_sub.get("items", {}).get("data", [])
    price_id: str = items[0]["price"]["id"] if items else ""
    tier = _price_id_to_tier(price_id)

    _gate.update_tier_from_stripe(
        user_id=user_id,
        tier=tier,
        stripe_customer_id=stripe_sub.get("customer", ""),
        stripe_subscription_id=stripe_sub.get("id", ""),
        is_active=stripe_sub.get("status") in ("active", "trialing"),
    )
    logger.info("User %s upgraded to tier=%s", user_id, tier.value)


def _handle_subscription_deleted(stripe_sub: dict) -> None:
    """Process ``customer.subscription.deleted``."""
    user_id: str | None = stripe_sub.get("metadata", {}).get("user_id")
    if not user_id:
        logger.warning("Subscription deletion missing user_id metadata; skipping.")
        return
    _gate.deactivate(user_id)
    logger.info("Subscription cancelled for user=%s; reverted to FREE caps.", user_id)


def _price_id_to_tier(price_id: str) -> SubscriptionTier:
    """Reverse-lookup a Stripe price ID to a :class:`SubscriptionTier`.

    Falls back to ``FREE`` for unrecognised price IDs.
    """
    for tier, pid in TIER_STRIPE_PRICE_ID.items():
        if pid and pid == price_id:
            return tier
    logger.warning("Unknown Stripe price ID '%s'; defaulting to FREE.", price_id)
    return SubscriptionTier.FREE
