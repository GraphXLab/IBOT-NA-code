"""Shared command-line and orchestration logic for main experiments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch

from .backend import run_ibot_seed
from .config import (
    DATASETS,
    DATA_ROOT,
    DEFAULT_HYPERPARAM_CONFIG,
    DEFAULT_IBOT_CODE_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_TIMEOUT_S,
    MODEL_NAME,
    RESULT_ROOT,
    TEMP_ROOT,
    canonical_dataset,
)
from .hyperparams import PARAMETER_KEYS, load_ibot_hyperparameters, parse_override
from .records import append_records, latest_records, read_jsonl
from .reporting import write_results_workbook
from .runner import execute_seed_group, is_oom_status


def _timeout(value: str) -> float:
    timeout = float(value)
    if timeout <= 0 or timeout > DEFAULT_TIMEOUT_S:
        raise argparse.ArgumentTypeError("timeout must be in (0, 3600]")
    return timeout


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--timeout-s", "--task-timeout-s", type=_timeout, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--temp-dir", type=Path, default=TEMP_ROOT)
    parser.add_argument("--result-file", type=Path, default=RESULT_ROOT / "main_results.xlsx")
    parser.add_argument("--hyperparams-config", type=Path, default=DEFAULT_HYPERPARAM_CONFIG)
    parser.add_argument("--ibot-code-root", type=Path, default=DEFAULT_IBOT_CODE_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--rerun", action="store_true", help="Append a new attempt even if a seed is recorded.")
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest without starting training.")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--gamma-p", "--gamma_p", dest="gamma_p", type=float)
    parser.add_argument(
        "--init-threshold-lambda", "--init_threshold_lambda", dest="init_threshold_lambda", type=float
    )
    parser.add_argument("--ib-kl-weight", "--ib_kl_weight", dest="ib_kl_weight", type=float)
    parser.add_argument("--ib-anchor-weight", "--ib_anchor_weight", dest="ib_anchor_weight", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="IBOT-NA model override. Later overrides take precedence.",
    )


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return value


def _explicit_overrides(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    parameter_overrides = {
        key: getattr(args, key)
        for key in PARAMETER_KEYS
        if getattr(args, key, None) is not None
    }
    extra_overrides: dict[str, Any] = {}
    for text in args.override:
        key, value = parse_override(text)
        if key in PARAMETER_KEYS:
            parameter_overrides[key] = value
        else:
            extra_overrides[key] = value
    return parameter_overrides, extra_overrides


def _write_manifest(temp_dir: Path, run_id: str, manifest: dict[str, Any]) -> None:
    manifest_dir = temp_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (manifest_dir / f"{run_id}.json").write_text(text, encoding="utf-8")
    (temp_dir / "last_run.json").write_text(text, encoding="utf-8")


def _refresh_main_results(records_path: Path, result_file: Path) -> None:
    """Rebuild the main workbook from every recorded canonical seed."""
    write_results_workbook(
        read_jsonl(records_path),
        result_file,
        planned_seeds=DEFAULT_SEEDS,
    )


def _ensure_inputs(args: argparse.Namespace) -> None:
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_root}")
    expected_count = len(DATASETS)
    if len(list(args.data_root.glob("*-pa_0.2.npz"))) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} NPZ datasets in {args.data_root}")
    if not args.hyperparams_config.is_file():
        raise FileNotFoundError(f"Hyperparameter JSON not found: {args.hyperparams_config}")
    if not (args.ibot_code_root / "main.py").is_file():
        raise FileNotFoundError(f"IBOT-NA program not found: {args.ibot_code_root / 'main.py'}")


def run_selected(
    args: argparse.Namespace,
    *,
    datasets: Iterable[str],
) -> int:
    canonical_datasets = tuple(dict.fromkeys(canonical_dataset(value) for value in datasets))
    if not canonical_datasets:
        raise ValueError("At least one dataset is required")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("--seeds must be non-empty and unique")
    args.python = args.python.resolve()
    args.data_root = args.data_root.resolve()
    args.temp_dir = args.temp_dir.resolve()
    args.result_file = args.result_file.resolve()
    args.hyperparams_config = args.hyperparams_config.resolve()
    args.ibot_code_root = args.ibot_code_root.resolve()
    _ensure_inputs(args)
    device = _device(args.device)
    parameter_overrides, extra_overrides = _explicit_overrides(args)
    hyperparams = load_ibot_hyperparameters(args.hyperparams_config)
    for params in hyperparams.values():
        params.update(
            {
                key: int(value) if key == "epochs" else float(value)
                for key, value in parameter_overrides.items()
            }
        )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.temp_dir / "runs" / run_id
    records_path = args.temp_dir / "experiment_records.jsonl"
    tasks = [
        {
            "dataset": dataset,
            "model": MODEL_NAME,
            "seeds": list(args.seeds),
            "params": {**hyperparams[dataset], **extra_overrides},
        }
        for dataset in canonical_datasets
    ]
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": list(canonical_datasets),
        "model": MODEL_NAME,
        "seeds": list(args.seeds),
        "device": device,
        "timeout_s": args.timeout_s,
        "split_protocol": "train10_val10_test80",
        "selection": "best_by_validation_mrr",
        "deterministic": True,
        "data_root": str(args.data_root),
        "result_file": str(args.result_file),
        "records_file": str(records_path),
        "hyperparams_config": str(args.hyperparams_config),
        "ibot_code_root": str(args.ibot_code_root),
        "parameter_overrides": parameter_overrides,
        "extra_overrides": extra_overrides,
        "tasks": tasks,
    }
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.temp_dir, run_id, manifest)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    existing = latest_records(read_jsonl(records_path))
    for task_index, task in enumerate(tasks, start=1):
        dataset = task["dataset"]
        model = task["model"]
        group_existing = [
            existing[(dataset, model, int(seed))]
            for seed in args.seeds
            if (dataset, model, int(seed)) in existing
        ]
        if not args.rerun and any(is_oom_status(row) for row in group_existing):
            print(f"SKIP {task_index}/{len(tasks)} dataset={dataset} model={model} previous=OOM", flush=True)
            continue
        pending_seeds = [
            int(seed)
            for seed in args.seeds
            if args.rerun or (dataset, model, int(seed)) not in existing
        ]
        if not pending_seeds:
            print(f"SKIP {task_index}/{len(tasks)} dataset={dataset} model={model} complete", flush=True)
            continue
        print(
            f"TASK {task_index}/{len(tasks)} dataset={dataset} model={model} seeds={pending_seeds}",
            flush=True,
        )

        def run_seed(seed: int) -> dict[str, Any]:
            task_root = run_root / "logs" / dataset / model / f"seed_{seed}"
            return run_ibot_seed(
                dataset=dataset,
                seed=seed,
                params=hyperparams[dataset],
                extra_overrides=extra_overrides,
                python=args.python,
                code_root=args.ibot_code_root,
                data_root=args.data_root,
                task_root=task_root,
                device=device,
                timeout_s=args.timeout_s,
            )

        rows = execute_seed_group(
            dataset=dataset,
            model=model,
            seeds=pending_seeds,
            run_seed=run_seed,
        )
        for row in rows:
            row["run_id"] = run_id
        append_records(records_path, rows)
        existing.update(latest_records(rows))
        _refresh_main_results(records_path, args.result_file)
        statuses = ",".join(str(row.get("status")) for row in rows)
        print(f"DONE dataset={dataset} model={model} statuses={statuses}", flush=True)

    _refresh_main_results(records_path, args.result_file)
    return 0
