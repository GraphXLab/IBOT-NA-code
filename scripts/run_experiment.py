"""Run IBOT-NA on selected datasets under the main protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ibot_na.experiments.cli import add_common_arguments, run_selected
from ibot_na.entrypoint import run_entrypoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", "--dataset", nargs="+", required=True)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_selected(args, datasets=args.datasets)


if __name__ == "__main__":
    run_entrypoint(main)
