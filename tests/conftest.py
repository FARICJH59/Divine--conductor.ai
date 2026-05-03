# pytest conftest for Divine Conductor AI
import sys
from pathlib import Path

# Ensure the src/ package tree is importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
