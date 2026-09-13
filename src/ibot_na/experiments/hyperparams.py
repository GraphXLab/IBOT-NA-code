"""Read per-dataset IBOT-NA hyperparameters from the package JSON config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import DATASETS, canonical_dataset


PARAMETER_KEYS = (
    "alpha",
    "gamma_p",
    "init_threshold_lambda",
    "ib_kl_weight",
    "ib_anchor_weight",
    "lr",
    "epochs",
)

def _number(value: Any, key: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Missing or invalid {key}")
    return float(value)


def _tuning_dataset_name(dataset: str) -> str:
    if dataset == "douban":
        return "Douban-pa"
    return f"{dataset}-pa"


def load_ibot_hyperparameters(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate all retained dataset parameters from JSON."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Hyperparameter JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid hyperparameter JSON: {path}: {error}") from error
    if "fixed_parameters" in payload and "best_parameters" in payload:
        fixed_parameters = payload.get("fixed_parameters")
        raw_datasets = payload.get("best_parameters")
        if not isinstance(fixed_parameters, dict):
            raise ValueError("Hyperparameter JSON fixed_parameters must be an object")
        if not isinstance(raw_datasets, dict):
            raise ValueError("Hyperparameter JSON best_parameters must be an object")
    else:
        if payload.get("schema_version") != "ibot-na-hyperparams-v1":
            raise ValueError("Unsupported hyperparameter schema_version")
        if payload.get("model") != "IBOT-NA":
            raise ValueError("Hyperparameter JSON model must be IBOT-NA")
        fixed_parameters = {}
        raw_datasets = payload.get("datasets")
        if not isinstance(raw_datasets, dict):
            raise ValueError("Hyperparameter JSON datasets must be an object")

    result: dict[str, dict[str, Any]] = {}
    for raw_dataset, values in raw_datasets.items():
        dataset = canonical_dataset(raw_dataset)
        if dataset in result:
            raise ValueError(f"Duplicate hyperparameter row for {dataset}")
        if not isinstance(values, dict):
            raise ValueError(f"{dataset}: parameters must be an object")
        merged_values = dict(fixed_parameters)
        merged_values.update(values)
        tuning_name = merged_values.get("tuning_dataset_name", _tuning_dataset_name(dataset))
        if not isinstance(tuning_name, str) or not tuning_name.strip():
            raise ValueError(f"{dataset}: invalid tuning_dataset_name")
        record: dict[str, Any] = {
            "dataset": dataset,
            "tuning_dataset_name": tuning_name,
            "model": "IBOT-NA",
        }
        for key in PARAMETER_KEYS:
            value = _number(merged_values.get(key), key)
            record[key] = int(value) if key == "epochs" else value
        result[dataset] = record

    missing_datasets = sorted(set(DATASETS) - set(result))
    extra_datasets = sorted(set(result) - set(DATASETS))
    if missing_datasets:
        raise ValueError("JSON has no IBOT-NA parameters for: " + ", ".join(missing_datasets))
    if extra_datasets:
        raise ValueError("JSON contains unsupported datasets: " + ", ".join(extra_datasets))
    return result


def resolve_ibot_hyperparameters(
    config_path: str | Path,
    dataset: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve JSON values, then apply explicit command-line overrides."""
    canonical = canonical_dataset(dataset)
    params = dict(load_ibot_hyperparameters(config_path)[canonical])
    for key, value in (overrides or {}).items():
        if key not in PARAMETER_KEYS:
            raise ValueError(f"Unsupported IBOT-NA parameter override: {key}")
        params[key] = int(value) if key == "epochs" else float(value)
    return params


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Override must use key=value: {text!r}")
    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override has an empty key: {text!r}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value
