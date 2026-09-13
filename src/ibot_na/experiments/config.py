"""Datasets, paths, and defaults for the IBOT-NA main experiment."""

from __future__ import annotations

import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PACKAGE_ROOT / "data"
TEMP_ROOT = PACKAGE_ROOT / "Temp"
RESULT_ROOT = PACKAGE_ROOT / "result"
CONFIG_ROOT = PACKAGE_ROOT / "config"
DEFAULT_HYPERPARAM_CONFIG = CONFIG_ROOT / "ibot_na_hyperparams.json"
DEFAULT_IBOT_CODE_ROOT = PACKAGE_ROOT / "models" / "IBOT-NA"
MODEL_NAME = "IBOT-NA"

DATASETS = (
    "foursquare-twitter",
    "phone-email",
    "arxiv",
    "GGI",
    "douban",
    "ACM-DBLP",
    "dbp15k_zh-en",
)

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TIMEOUT_S = 3600.0
RESULT_COLUMNS = ("Model", "Seeds", "Hits@1", "Hits@10", "MRR", "Time(s)", "Mem.(GB)", "Status")


def _token(value: str) -> str:
    token = str(value).strip().replace("_", "-").casefold()
    token = re.sub(r"-pa(?:-?0\.2)?$", "", token)
    token = re.sub(r"-0\.2$", "", token)
    return token


_DATASET_LOOKUP = {_token(name): name for name in DATASETS}


def canonical_dataset(value: str) -> str:
    token = _token(value)
    try:
        return _DATASET_LOOKUP[token]
    except KeyError as error:
        raise ValueError(f"Unknown dataset {value!r}. Choose from: {', '.join(DATASETS)}") from error

