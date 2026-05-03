"""GenreFactory — loads and caches named genre definitions from YAML files.

YAML files are stored in ``config/genres/<key>.yaml`` (relative to the
working directory, or in a custom directory supplied at construction time).
Each file's stem is the genre key.  The factory is the single source of
truth for all available genres.

Usage::

    factory = GenreFactory()
    genre = factory.load("cyber_noir")          # single genre
    all_genres = factory.load_all()             # dict[key → Genre]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from divine_conductor.models.production import Genre

logger = logging.getLogger(__name__)

# Default location: config/genres/ relative to the current working directory
_DEFAULT_GENRES_DIR = Path("config") / "genres"


class GenreFactory:
    """Discovers and loads :class:`~divine_conductor.models.production.Genre`
    objects from YAML files on disk.

    Args:
        genres_dir: Directory that contains the genre YAML files.
            Defaults to ``config/genres/`` relative to the current
            working directory.
    """

    def __init__(self, genres_dir: Path | str | None = None) -> None:
        self._genres_dir = Path(genres_dir) if genres_dir is not None else _DEFAULT_GENRES_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, key: str) -> Genre:
        """Load a single genre by key.

        Args:
            key: Genre key matching a ``<key>.yaml`` file in the genres
                directory (e.g. ``"cyber_noir"``).

        Returns:
            The parsed :class:`Genre` object.

        Raises:
            FileNotFoundError: If no matching YAML file exists.
            ValueError: If the file is missing required fields.
        """
        path = self._genres_dir / f"{key}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Genre '{key}' not found. Expected file: {path}"
            )
        return self._load_file(path)

    def load_all(self) -> dict[str, Genre]:
        """Load every genre YAML in the genres directory.

        Returns:
            Mapping of genre key → :class:`Genre`.  Empty dict if the
            directory does not exist or contains no YAML files.
        """
        if not self._genres_dir.exists():
            logger.warning("Genres directory not found: %s", self._genres_dir)
            return {}

        genres: dict[str, Genre] = {}
        for path in sorted(self._genres_dir.glob("*.yaml")):
            try:
                genre = self._load_file(path)
                genres[genre.key] = genre
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load genre from %s: %s", path, exc)
        return genres

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(path: Path) -> Genre:
        """Parse a single genre YAML file into a :class:`Genre`."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        key = raw.get("key") or path.stem
        visual_anchors: list[str] = [str(a) for a in raw.get("visual_anchors", [])]
        lighting: str = str(raw.get("lighting", ""))
        camera_tech: str = str(raw.get("camera_tech", ""))
        wardrobe_modifier: str = str(raw.get("wardrobe_modifier", ""))

        return Genre(
            key=key,
            visual_anchors=visual_anchors,
            lighting=lighting,
            camera_tech=camera_tech,
            wardrobe_modifier=wardrobe_modifier,
        )
