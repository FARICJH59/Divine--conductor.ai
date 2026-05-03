"""Divine Conductor AI — CLI module.

This module contains the ``main()`` entrypoint registered as the
``divine-conductor`` console script by ``pyproject.toml``.

It can also be run directly::

    python -m divine_conductor.cli --config config/example_pipeline.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

from divine_conductor.models.production import PalettePreset, ProductionConfig
from divine_conductor.pipeline.orchestrator import PipelineOrchestrator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divine-conductor",
        description=(
            "Agentic AI film production pipeline for high-fidelity biblical "
            "storytelling with Veo 3.1 consistency logic."
        ),
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        metavar="FILE",
        help="Path to a YAML pipeline configuration file.",
    )
    source.add_argument(
        "--passage",
        metavar="TEXT",
        help="Inline scripture / screenplay passage to produce.",
    )

    parser.add_argument(
        "--name",
        default="Untitled Production",
        help="Production title (used with --passage).",
    )
    parser.add_argument(
        "--style",
        choices=["cinematic", "documentary", "animated"],
        default="cinematic",
        help="Visual style preset (default: cinematic).",
    )
    parser.add_argument(
        "--palette",
        choices=[p.value for p in PalettePreset],
        default=PalettePreset.WARM_GOLDEN_DAWN.value,
        help="Colour palette preset (default: warm_golden_dawn).",
    )
    parser.add_argument(
        "--output",
        default="output",
        metavar="DIR",
        help="Output directory for generated shot bundles (default: output/).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "txt"],
        default="json",
        help="Output serialisation format (default: json).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.config:
        orchestrator = PipelineOrchestrator.from_yaml(args.config)
    else:
        config = ProductionConfig(
            name=args.name,
            passage_text=args.passage,
            style=args.style,
            palette=PalettePreset(args.palette),
            output_format=args.format,
            output_path=args.output,
        )
        orchestrator = PipelineOrchestrator(config)

    state = orchestrator.run()
    summary = state.summary()
    print(
        f"\n✅  Done — {summary['shots']} shots across "
        f"{summary['scenes']} scenes "
        f"({summary['total_duration_seconds']:.1f}s total).\n"
        f"   Consistency issues: {summary['consistency_issues']}\n"
        f"   Output written to:  {state.config.output_path}/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
