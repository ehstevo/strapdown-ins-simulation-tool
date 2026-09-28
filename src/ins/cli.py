"""Command-line entry point for the strapdown INS Monte Carlo simulator."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ins.config import ConfigurationError, load_config
from ins.monte_carlo import run_monte_carlo
from ins.reporting import save_results


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a configuration-driven strapdown INS Monte Carlo simulation."
    )
    parser.add_argument("config", type=Path, help="Path to a TOML simulation configuration.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output directory specified by the configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the simulator from the command line."""

    arguments = _arguments()
    try:
        config = load_config(arguments.config)
    except ConfigurationError as error:
        raise SystemExit(f"Configuration error: {error}") from error

    if arguments.output_dir is not None:
        config = replace(config, output_dir=arguments.output_dir)

    print(f"Generating {config.scenario.name} truth trajectory...")
    print(f"Propagating {config.runs} Monte Carlo runs with seed {config.seed}...")
    result = run_monte_carlo(config)
    output_dir = save_results(result, config)
    print(f"Wrote report and ensemble statistics to {output_dir}")


if __name__ == "__main__":
    main()
