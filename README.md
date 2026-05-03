# 🎬 Divine Conductor AI

> **Agentic AI film production pipeline for high-fidelity biblical storytelling with Veo 3.1 consistency logic.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Divine Conductor AI is an end-to-end agentic film production pipeline purpose-built for producing cinematic, high-fidelity biblical narratives. It chains together specialised AI agents — a **Director**, a **Cinematographer**, and a **Narrator** — that collaborate autonomously to turn a scriptural passage into a production-ready sequence of video prompts, guided by a **Veo 3.1 consistency engine** that preserves character appearance, lighting mood, and scene continuity across every generated shot.

```
Scripture / Script Input
        │
        ▼
┌───────────────────┐
│   NarratorAgent   │  ─── breaks passage into scenes with dialogue & context
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   DirectorAgent   │  ─── assigns camera angles, pacing, emotional arc
└────────┬──────────┘
         │
         ▼
┌──────────────────────────┐
│  CinematographerAgent    │  ─── builds per-shot visual prompts
└────────┬─────────────────┘
         │
         ▼
┌───────────────────────────┐
│  Veo3Consistency Engine   │  ─── anchors character IDs, palette, & continuity
└────────┬──────────────────┘
         │
         ▼
 Final Shot Prompt Bundle  ──► Veo 3.1 / downstream render pipeline
```

---

## Features

| Feature | Description |
|---|---|
| **Agentic orchestration** | Director, Cinematographer and Narrator agents run autonomously via a configurable pipeline |
| **Veo 3.1 consistency logic** | Character embeddings, colour palette anchors, and cross-shot continuity checks |
| **Biblical storytelling focus** | Pre-built prompt templates for Old & New Testament passages |
| **YAML-driven configuration** | Fully declarative pipeline config — no code changes needed to run a new story |
| **Extensible agent framework** | Drop in custom agents by subclassing `BaseAgent` |

---

## Quick Start

### Prerequisites

```bash
python -m pip install -r requirements.txt
```

### Run the example pipeline

```bash
python main.py --config config/example_pipeline.yaml
```

### Run a single scripture passage inline

```bash
python main.py --passage "In the beginning God created the heavens and the earth." \
               --style cinematic --output output/genesis_1_1
```

---

## Project Structure

```
divine-conductor-ai/
├── main.py                          # CLI entrypoint
├── requirements.txt
├── pyproject.toml
├── config/
│   └── example_pipeline.yaml        # Example pipeline configuration
├── src/divine_conductor/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── production.py            # Scene, Character, Shot, ProductionState models
│   ├── consistency/
│   │   ├── __init__.py
│   │   └── veo_consistency.py       # Veo 3.1 consistency engine
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseAgent abstract class
│   │   ├── narrator.py              # NarratorAgent
│   │   ├── director.py              # DirectorAgent
│   │   └── cinematographer.py       # CinematographerAgent
│   └── pipeline/
│       ├── __init__.py
│       └── orchestrator.py          # Pipeline orchestrator
└── tests/
    ├── test_models.py
    ├── test_consistency.py
    └── test_pipeline.py
```

---

## Configuration Reference

```yaml
# config/example_pipeline.yaml
pipeline:
  name: "Genesis Creation Sequence"
  style: cinematic          # cinematic | documentary | animated
  aspect_ratio: "16:9"
  fps: 24

passage:
  book: Genesis
  chapter: 1
  verses: "1-5"
  text: |
    In the beginning God created the heavens and the earth.
    ...

characters:
  - id: narrator_voice
    description: "Warm, authoritative off-screen narrator"

consistency:
  palette: warm_golden_dawn
  anchor_shots: true
  character_id_strength: 0.85

output:
  format: json              # json | yaml | txt
  path: output/genesis_1
```

---

## Architecture

### Agents

Each agent receives a `ProductionState` object and returns an enriched version of it.

| Agent | Responsibility |
|---|---|
| `NarratorAgent` | Parses scripture into discrete `Scene` objects with dialogue, setting, and emotional tone |
| `DirectorAgent` | Annotates each `Scene` with camera work, pacing beats, and director's notes |
| `CinematographerAgent` | Converts annotated scenes into concrete `Shot` objects with full Veo-compatible text prompts |

### Veo 3.1 Consistency Engine

The `Veo3ConsistencyEngine` maintains:

- **CharacterAnchor** – a canonical visual description for each named character, injected into every shot prompt that features them.
- **PaletteAnchor** – a locked colour grade (e.g. `warm_golden_dawn`) applied consistently across shots.
- **ContinuityChecker** – detects and flags prompt contradictions between consecutive shots (costume changes, time-of-day jumps, etc.).

---

## License

MIT © 2025 FARICJH59