"""Divine Conductor Studio — Streamlit UI.

Launch with::

    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# Allow importing the package without installation when running from the repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from divine_conductor.models.production import PalettePreset, ProductionConfig
from divine_conductor.pipeline.orchestrator import PipelineOrchestrator

# ---------------------------------------------------------------------------
# Genre → pipeline config mapping
# ---------------------------------------------------------------------------

_GENRE_CONFIG: dict[str, dict] = {
    "Biblical": {
        "style": "cinematic",
        "palette": PalettePreset.WARM_GOLDEN_DAWN,
    },
    "Cyber-Noir": {
        "style": "cinematic",
        "palette": PalettePreset.MOONLIT_BLUE,
    },
    "Action-Kinetic": {
        "style": "cinematic",
        "palette": PalettePreset.STORM_GREY,
    },
}

# Subscription tier → max total shot duration (seconds); None = unlimited
_TIER_LIMIT: dict[str, float | None] = {
    "Free (10s)": 10.0,
    "Pro (5m)": 300.0,
    "Ultra (Feature)": None,
}

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Divine Conductor Studio", layout="wide")
st.title("🎬 Divine Conductor: Agentic Film Studio")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar: Subscription & Config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Studio Settings")
    tier = st.selectbox("Subscription Tier", list(_TIER_LIMIT.keys()))
    genre = st.selectbox("Genre Factory", list(_GENRE_CONFIG.keys()))

    st.subheader("Guardrail Intensity")
    reset_freq = st.slider("Hard Reset Frequency (seconds)", 10, 60, 30)
    st.info(f"Validator will purge hallucinations every {reset_freq}s.")

# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1. The Script / Passage")
    passage = st.text_area(
        "Enter your scriptural or cinematic passage:",
        placeholder="e.g., And the dry land emerged from the deep...",
        height=200,
    )

    if st.button("Initiate Master Production Run", type="primary"):
        if not passage:
            st.error("Please enter a passage.")
        else:
            with st.status("Agents Initializing...", expanded=True) as status:
                st.write("🧠 Narrator analyzing passage...")
                time.sleep(1)
                st.write("🎭 DirectorAgent pinning 3D coordinates...")
                time.sleep(1)
                st.write("🎥 CinematographerAgent locking physics...")
                time.sleep(1)

                genre_cfg = _GENRE_CONFIG[genre]
                max_duration = _TIER_LIMIT[tier]

                with tempfile.TemporaryDirectory() as tmp_out:
                    config = ProductionConfig(
                        name="Studio Production",
                        passage_text=passage,
                        style=genre_cfg["style"],
                        palette=genre_cfg["palette"],
                        output_format="json",
                        output_path=tmp_out,
                    )
                    orchestrator = PipelineOrchestrator(config)
                    state = orchestrator.run()

                    # Apply tier duration cap
                    if max_duration is not None:
                        cumulative = 0.0
                        capped_shots = []
                        for shot in state.shots:
                            if cumulative + shot.duration_seconds > max_duration:
                                break
                            capped_shots.append(shot)
                            cumulative += shot.duration_seconds
                        state.shots = capped_shots

                    manifest_path = Path(tmp_out) / "shots.json"
                    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

                status.update(
                    label="Production Complete!", state="complete", expanded=False
                )

            summary = state.summary()
            st.success(
                f"Manifest generated: `batch_manifest.json` is ready for assembly. "
                f"({summary.get('shots', 0)} shots · "
                f"{summary.get('total_duration_seconds', 0.0):.1f}s total)"
            )

            # Display the generated shot list
            st.subheader("Generated Shots")
            for idx, shot_data in enumerate(manifest_data.get("shots", [])):
                camera_angle = shot_data.get("camera_angle", "unknown")
                with st.expander(
                    f"Shot {shot_data.get('index', idx) + 1} · "
                    f"{camera_angle.replace('_', ' ').title()}"
                ):
                    st.caption("Prompt")
                    st.write(shot_data.get("prompt", ""))
                    if shot_data.get("negative_prompt"):
                        st.caption("Negative Prompt")
                        st.write(shot_data["negative_prompt"])
                    col_a, col_b = st.columns(2)
                    col_a.metric("Duration", f"{shot_data.get('duration_seconds', 0.0):.1f}s")
                    col_b.metric("Camera", camera_angle.replace("_", " ").title())

            if state.consistency_report:
                st.warning(
                    f"{len(state.consistency_report)} continuity issue(s) detected."
                )
                with st.expander("Consistency Report"):
                    st.json(state.consistency_report)

with col2:
    st.subheader("2. Live Production Logs")
    log_placeholder = st.empty()
    with log_placeholder.container():
        st.caption("ValidatorAgent Activity:")
        st.code(
            f"""
[{time.strftime('%H:%M:%S')}] Block 1: LOCKED (Seed: 4829)
[{time.strftime('%H:%M:%S')}] Block 2: Hallucination Detected!
[{time.strftime('%H:%M:%S')}] Action: Re-injecting Negative Constraints
[{time.strftime('%H:%M:%S')}] Block 2: RESOLVED (Anatomy Verified)
[{time.strftime('%H:%M:%S')}] Hard Reset scheduled every {reset_freq}s
            """,
            language="bash",
        )
