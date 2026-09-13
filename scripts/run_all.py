"""Run IBOT-NA on all seven main-experiment datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ibot_na.experiments.cli import add_common_arguments, run_selected
from ibot_na.experiments.config import DATASETS
from ibot_na.entrypoint import run_entrypoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> int:
    return run_selected(parse_args(), datasets=DATASETS)


if __name__ == "__main__":
    run_entrypoint(main)
