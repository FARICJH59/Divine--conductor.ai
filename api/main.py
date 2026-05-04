"""Divine Conductor AI — FastAPI web interface.

Exposes the agentic pipeline and Stripe subscription gate via HTTP so the
studio can be launched with a single ``docker-compose up`` command.

Endpoints
---------
GET  /health                          — liveness probe
POST /api/run                         — run the pipeline synchronously
POST /api/checkout                    — create a Stripe Checkout Session
POST /api/webhook                     — handle Stripe webhook events
GET  /api/subscription/{user_id}      — query active subscription tier
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── Ensure the project source is importable when run outside of an install ──
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import stripe  # type: ignore[import]
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from divine_conductor.models.production import (
    Character,
    PalettePreset,
    ProductionConfig,
)
from divine_conductor.pipeline.orchestrator import PipelineOrchestrator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("divine_conductor.api")

# ---------------------------------------------------------------------------
# Stripe configuration
# ---------------------------------------------------------------------------

stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
_STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
_STRIPE_SUCCESS_URL = os.getenv("STRIPE_SUCCESS_URL", "http://localhost:8000/checkout/success")
_STRIPE_CANCEL_URL = os.getenv("STRIPE_CANCEL_URL", "http://localhost:8000/checkout/cancel")

# ---------------------------------------------------------------------------
# In-memory subscription store (replace with a real DB in production)
# ---------------------------------------------------------------------------

_SUBSCRIPTIONS: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Divine Conductor AI",
    description="Agentic multimodal film studio — pipeline API & subscription gate.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CharacterSchema(BaseModel):
    id: str
    name: str = ""
    description: str
    role: str = "supporting"


class RunRequest(BaseModel):
    passage_text: str = Field(..., min_length=1, description="Scriptural or screenplay text to produce.")
    name: str = Field("Divine Conductor Production", description="Human-readable production title.")
    style: str = Field("cinematic", description="Visual style: cinematic | documentary | animated.")
    aspect_ratio: str = Field("16:9")
    fps: int = Field(24, ge=1, le=120)
    palette: str = Field("warm_golden_dawn", description="Colour palette preset.")
    character_id_strength: float = Field(0.85, ge=0.0, le=1.0)
    anchor_shots: bool = True
    output_format: str = Field("json", description="Output format: json | yaml | txt.")
    characters: list[CharacterSchema] = Field(default_factory=list)


class CheckoutRequest(BaseModel):
    user_id: str = Field(..., description="Your application's user identifier.")
    tier: str = Field("pro", description="Subscription tier: pro | ultra.")
    email: str = Field("", description="Customer e-mail pre-fill (optional).")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe used by Docker Compose and load balancers."""
    return {"status": "ok", "service": "divine-conductor-ai"}


@app.post("/api/run", tags=["pipeline"])
def run_pipeline(req: RunRequest) -> JSONResponse:
    """Run the full agentic pipeline synchronously and return the shot bundle.

    For production use-cases with long passages, consider wrapping this in an
    async task queue (e.g. Celery + Redis) so the HTTP request does not time out.
    """
    try:
        palette = PalettePreset(req.palette)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown palette '{req.palette}'. "
            f"Valid values: {[p.value for p in PalettePreset]}",
        )

    characters = [
        Character(
            id=c.id,
            name=c.name or c.id,
            description=c.description,
            role=c.role,
        )
        for c in req.characters
    ]

    output_dir = os.getenv("OUTPUT_DIR", "output")

    config = ProductionConfig(
        name=req.name,
        passage_text=req.passage_text,
        style=req.style,
        aspect_ratio=req.aspect_ratio,
        fps=req.fps,
        palette=palette,
        character_id_strength=req.character_id_strength,
        anchor_shots=req.anchor_shots,
        output_format=req.output_format,
        output_path=output_dir,
        characters=characters,
    )

    orchestrator = PipelineOrchestrator(config)
    state = orchestrator.run()

    shots_payload = [
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
    ]

    return JSONResponse(
        content={
            "production": state.config.name,
            "summary": state.summary(),
            "consistency_report": state.consistency_report,
            "shots": shots_payload,
        }
    )


@app.post("/api/checkout", tags=["subscription"])
def create_checkout(req: CheckoutRequest) -> JSONResponse:
    """Create a Stripe Checkout Session for a subscription upgrade.

    Returns the Stripe-hosted payment URL that the client should redirect to.
    """
    if not stripe.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe is not configured. Set STRIPE_SECRET_KEY in your environment.",
        )

    # Price lookup key follows the convention: divine_conductor_<tier>_monthly
    price_lookup_key = f"divine_conductor_{req.tier.lower()}_monthly"

    try:
        session_params: dict[str, Any] = {
            "mode": "subscription",
            "line_items": [{"price": price_lookup_key, "quantity": 1}],
            "success_url": _STRIPE_SUCCESS_URL + "?session_id={CHECKOUT_SESSION_ID}",
            "cancel_url": _STRIPE_CANCEL_URL,
            "metadata": {"user_id": req.user_id, "tier": req.tier},
        }
        if req.email:
            session_params["customer_email"] = req.email

        session = stripe.checkout.Session.create(**session_params)
    except stripe.error.InvalidRequestError as exc:
        logger.error("Stripe InvalidRequestError: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except stripe.error.StripeError as exc:
        logger.error("Stripe error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment provider error. Please try again.",
        )

    return JSONResponse(content={"checkout_url": session.url, "session_id": session.id})


@app.post("/api/webhook", tags=["subscription"])
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(default="", alias="stripe-signature"),
) -> JSONResponse:
    """Receive and verify Stripe webhook events.

    Stripe sends signed POST requests to this endpoint when subscription
    events occur (e.g. payment succeeded, subscription cancelled).
    """
    payload = await request.body()

    if not _STRIPE_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Webhook endpoint is disabled: STRIPE_WEBHOOK_SECRET is not configured.",
        )

    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, _STRIPE_WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Stripe webhook signature.",
        )

    _handle_stripe_event(event)
    return JSONResponse(content={"received": True})


def _handle_stripe_event(event: Any) -> None:
    """Update the in-memory subscription store based on the Stripe event type."""
    event_type: str = event["type"]
    data_object = event["data"]["object"]

    if event_type == "checkout.session.completed":
        user_id = data_object.get("metadata", {}).get("user_id")
        tier = data_object.get("metadata", {}).get("tier", "pro")
        if user_id:
            _SUBSCRIPTIONS[user_id] = {
                "tier": tier,
                "status": "active",
                "stripe_customer_id": data_object.get("customer"),
                "stripe_subscription_id": data_object.get("subscription"),
                "activated_at": int(time.time()),
            }
            logger.info("Subscription activated for user %s (tier=%s).", user_id, tier)

    elif event_type in ("customer.subscription.deleted", "customer.subscription.paused"):
        subscription_id = data_object.get("id")
        for user_id, sub in _SUBSCRIPTIONS.items():
            if sub.get("stripe_subscription_id") == subscription_id:
                sub["status"] = "inactive"
                logger.info("Subscription deactivated for user %s.", user_id)
                break

    elif event_type == "customer.subscription.updated":
        subscription_id = data_object.get("id")
        new_status = data_object.get("status", "active")
        for sub in _SUBSCRIPTIONS.values():
            if sub.get("stripe_subscription_id") == subscription_id:
                sub["status"] = new_status
                break

    else:
        logger.debug("Unhandled Stripe event type: %s", event_type)


@app.get("/api/subscription/{user_id}", tags=["subscription"])
def get_subscription(user_id: str) -> JSONResponse:
    """Return the active subscription record for a given user ID.

    Returns 404 if no subscription is found in the store (free tier).
    """
    # Sanitise user_id — only allow alphanumeric, hyphens, underscores
    if not all(c.isalnum() or c in "-_" for c in user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id may only contain alphanumeric characters, hyphens, and underscores.",
        )

    sub = _SUBSCRIPTIONS.get(user_id)
    if sub is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"user_id": user_id, "tier": "free", "status": "inactive"},
        )

    return JSONResponse(content={"user_id": user_id, **sub})
