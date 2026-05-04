"""Tests for the FastAPI subscription endpoints.

Uses ``httpx`` (bundled with ``fastapi[testclient]``) via Starlette's
``TestClient`` to exercise the endpoints in-process without a live Stripe
connection.  Stripe API calls are monkey-patched so no real credentials are
required.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from divine_conductor.models.user import SubscriptionTier


# ---------------------------------------------------------------------------
# Import the FastAPI app — stripe is stubbed before it runs any module-level
# code that hits the Stripe API.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Return a TestClient with a clean in-memory store for each module run."""
    # Patch stripe.api_key validation so we can import without real credentials
    with patch.dict("os.environ", {
        "STRIPE_SECRET_KEY": "sk_test_placeholder",
        "STRIPE_WEBHOOK_SECRET": "whsec_placeholder",
    }):
        from api.main import app, _gate
        # Reset the shared gate store before tests run
        _gate._store.clear()
        yield TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/subscription/{user_id}
# ---------------------------------------------------------------------------


class TestGetSubscription:
    def test_unknown_user_returns_free(self, client):
        resp = client.get("/api/subscription/unknown_user_abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "free"
        assert data["is_active"] is True
        assert data["max_duration_seconds"] == 10.0

    def test_returns_correct_tier_after_upsert(self, client):
        from api.main import _gate
        from divine_conductor.models.user import UserSubscription
        _gate.upsert(UserSubscription(user_id="pro_user", tier=SubscriptionTier.PRO))

        resp = client.get("/api/subscription/pro_user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "pro"
        assert data["max_duration_seconds"] == 300.0


# ---------------------------------------------------------------------------
# POST /api/checkout
# ---------------------------------------------------------------------------


class TestCheckout:
    def test_free_tier_rejected(self, client):
        resp = client.post(
            "/api/checkout", json={"user_id": "u1", "tier": "free"}
        )
        assert resp.status_code == 400
        assert "no payment" in resp.json()["detail"].lower()

    def test_pro_checkout_creates_session(self, client):
        mock_session = SimpleNamespace(
            url="https://checkout.stripe.com/session123",
            id="cs_test_123",
        )
        with patch("stripe.checkout.Session.create", return_value=mock_session):
            resp = client.post(
                "/api/checkout", json={"user_id": "u2", "tier": "pro"}
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["checkout_url"] == "https://checkout.stripe.com/session123"
        assert data["session_id"] == "cs_test_123"

    def test_ultra_checkout_creates_session(self, client):
        mock_session = SimpleNamespace(
            url="https://checkout.stripe.com/ultra456",
            id="cs_ultra_456",
        )
        with patch("stripe.checkout.Session.create", return_value=mock_session):
            resp = client.post(
                "/api/checkout", json={"user_id": "u3", "tier": "ultra"}
            )
        assert resp.status_code == 201
        assert resp.json()["session_id"] == "cs_ultra_456"

    def test_stripe_error_returns_502(self, client):
        import stripe as _stripe
        with patch(
            "stripe.checkout.Session.create",
            side_effect=_stripe.StripeError("Network error"),
        ):
            resp = client.post(
                "/api/checkout", json={"user_id": "u4", "tier": "pro"}
            )
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /api/webhook
# ---------------------------------------------------------------------------


def _build_webhook_event(event_type: str, sub_data: dict) -> dict:
    return {"type": event_type, "data": {"object": sub_data}}


class TestWebhook:
    def _post_webhook(self, client, event: dict) -> "Response":
        """Post a webhook event bypassing Stripe signature verification."""
        with patch(
            "stripe.Webhook.construct_event",
            return_value=event,
        ):
            return client.post(
                "/api/webhook",
                content=json.dumps(event).encode(),
                headers={"stripe-signature": "t=1,v1=fake"},
            )

    def test_subscription_created_updates_tier(self, client):
        from api.main import _gate
        event = _build_webhook_event(
            "customer.subscription.created",
            {
                "id": "sub_001",
                "customer": "cus_001",
                "status": "active",
                "metadata": {"user_id": "webhook_user_1", "tier": "pro"},
                "items": {
                    "data": [{"price": {"id": "price_pro_placeholder"}}]
                },
            },
        )
        resp = self._post_webhook(client, event)
        assert resp.status_code == 200
        assert resp.json()["received"] is True

        sub = _gate.get_subscription("webhook_user_1")
        assert sub.tier == SubscriptionTier.PRO
        assert sub.stripe_customer_id == "cus_001"
        assert sub.is_active is True

    def test_subscription_deleted_deactivates(self, client):
        from api.main import _gate
        from divine_conductor.models.user import UserSubscription
        _gate.upsert(
            UserSubscription(
                user_id="webhook_user_2",
                tier=SubscriptionTier.ULTRA,
                stripe_customer_id="cus_002",
                stripe_subscription_id="sub_002",
            )
        )
        event = _build_webhook_event(
            "customer.subscription.deleted",
            {
                "id": "sub_002",
                "customer": "cus_002",
                "status": "canceled",
                "metadata": {"user_id": "webhook_user_2"},
                "items": {"data": []},
            },
        )
        resp = self._post_webhook(client, event)
        assert resp.status_code == 200
        assert _gate.get_subscription("webhook_user_2").is_active is False

    def test_invalid_signature_returns_400(self, client):
        import stripe as _stripe
        with patch(
            "stripe.Webhook.construct_event",
            side_effect=_stripe.SignatureVerificationError("bad", "sig"),
        ):
            resp = client.post(
                "/api/webhook",
                content=b"{}",
                headers={"stripe-signature": "bad"},
            )
        assert resp.status_code == 400

    def test_unknown_event_is_ignored(self, client):
        event = {"type": "payment_intent.created", "data": {"object": {}}}
        resp = self._post_webhook(client, event)
        assert resp.status_code == 200
