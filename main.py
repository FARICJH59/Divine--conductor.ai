#!/usr/bin/env python3
"""Divine Conductor AI — root launcher.

This script delegates to :mod:`divine_conductor.cli` so that the project can
be run directly from the repository root without installation::

    python main.py --config config/example_pipeline.yaml
    python main.py --passage "In the beginning..." --style cinematic

When the package is installed via pip the ``divine-conductor`` console script
(registered in ``pyproject.toml``) calls :func:`divine_conductor.cli.main`
directly, bypassing this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when running directly (without pip install)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from divine_conductor.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
