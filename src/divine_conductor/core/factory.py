"""GenreFactory — dynamically loads all YAML genre configs from a directory.

Any ``.yaml`` file placed in ``config/genres/`` is automatically registered
and becomes selectable as a ``--genre`` flag value at the CLI.

Example::

    factory = GenreFactory()
    dna = factory.get_genre_dna("noir")
    # {"genre": "noir", "vibe": "...", "lighting": "...", "camera_tech": "..."}
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default genres directory relative to the installed package tree.
# Callers can override by passing an explicit ``config_path``.
_DEFAULT_GENRES_DIR = Path(__file__).parent.parent.parent.parent / "config" / "genres"

GenreDNA = dict[str, Any]


class GenreFactory:
    """Registry of all genre director-DNA configs found in *config_path*.

    The factory scans *config_path* at construction time and loads every
    ``*.yaml`` file into an in-memory registry keyed by the file's stem
    (i.e. the filename without the ``.yaml`` extension).

    Args:
        config_path: Directory containing genre YAML files.  Defaults to
            ``config/genres/`` resolved relative to the package root so the
            factory works regardless of the current working directory.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else _DEFAULT_GENRES_DIR
        self.registry: dict[str, GenreDNA] = self._load_all_genres()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_genre_dna(self, genre_name: str) -> GenreDNA:
        """Return the DNA dict for *genre_name*, falling back to "biblical".

        Args:
            genre_name: Name of the desired genre (matches the YAML file stem).

        Returns:
            A dict with at least the keys ``genre``, ``vibe``, ``lighting``,
            and ``camera_tech``.  Falls back to the ``biblical`` entry if
            *genre_name* is not found.
        """
        if genre_name in self.registry:
            return self.registry[genre_name]
        if not self.registry:
            logger.warning(
                "Genre registry is empty; cannot resolve genre '%s'.", genre_name
            )
            return {}
        logger.warning(
            "Genre '%s' not found in registry %s; falling back to 'biblical'.",
            genre_name,
            list(self.registry.keys()),
        )
        return self.registry.get("biblical", next(iter(self.registry.values())))

    @property
    def available_genres(self) -> list[str]:
        """Sorted list of genre names currently in the registry."""
        return sorted(self.registry.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all_genres(self) -> dict[str, GenreDNA]:
        """Scan *config_path* and load every ``*.yaml`` file."""
        registry: dict[str, GenreDNA] = {}

        if not self.config_path.is_dir():
            logger.warning(
                "Genre config directory '%s' does not exist; registry will be empty.",
                self.config_path,
            )
            return registry

        for entry in os.scandir(self.config_path):
            if entry.is_file() and entry.name.endswith(".yaml"):
                name = entry.name[: -len(".yaml")]
                try:
                    with open(entry.path, encoding="utf-8") as fh:
                        data = yaml.safe_load(fh)
                    if isinstance(data, dict):
                        registry[name] = data
                        logger.debug("Loaded genre '%s' from %s", name, entry.path)
                    else:
                        logger.warning(
                            "Skipping '%s': expected a YAML mapping, got %s.",
                            entry.path,
                            type(data).__name__,
                        )
                except Exception as exc:  # pragma: no cover
                    logger.error("Failed to load genre config '%s': %s", entry.path, exc)

        logger.info(
            "GenreFactory: loaded %d genre(s): %s",
            len(registry),
            sorted(registry.keys()),
        )
        return registry
