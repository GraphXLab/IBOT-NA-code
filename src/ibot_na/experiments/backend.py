"""Subprocess execution for IBOT-NA main experiments."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import psutil

from ibot_na.data import load_dataset

from .config import TEMP_ROOT
from .hyperparams import PARAMETER_KEYS


BEST_VAL_RE = re.compile(
    r"BEST_VAL Epoch (?P<best_epoch>\d+), Val Hits@1: (?P<val_hits1>[0-9.eE+-]+), "
    r"Val Hits@10: (?P<val_hits10>[0-9.eE+-]+), Val MRR: (?P<val_mrr>[0-9.eE+-]+), "
    r"Test Hits@1: (?P<hits1>[0-9.eE+-]+), Test Hits@10: (?P<hits10>[0-9.eE+-]+), "
    r"Test MRR: (?P<mrr>[0-9.eE+-]+)"
)


def runtime_cache_root(task_root: Path, *, dataset: str, seed: int) -> Path:
    """Return a package-local cache path that stays below Windows MAX_PATH."""
    identity = f"{Path(task_root).resolve()}|{dataset}|{int(seed)}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return TEMP_ROOT / "runtime_cache" / digest


def _stringify(value: Any) -> str:
    return format(value, ".12g") if isinstance(value, float) else str(value)


def _gpu_memory_gb(pid: int) -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    maximum = None
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or int(fields[0]) != pid:
            continue
        try:
            value = float(fields[1]) / 1024.0
        except ValueError:
            continue
        maximum = max(maximum or 0.0, value)
    return maximum


def _classify_failure(stdout_path: Path, stderr_path: Path, return_code: int) -> tuple[str, str]:
    text = "\n".join(
        (
            stdout_path.read_text(encoding="utf-8", errors="replace"),
            stderr_path.read_text(encoding="utf-8", errors="replace"),
        )
    )
    lowered = text.casefold()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "OOM", "CUDA out of memory"
    last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    return "FAILED", last_line[-1000:] or f"Process returned {return_code}"


def _parse_ibot_metrics(stdout_path: Path) -> dict[str, Any]:
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    matches = list(BEST_VAL_RE.finditer(text))
    if not matches:
        raise ValueError("BEST_VAL metrics not found in IBOT-NA output")
    values = matches[-1].groupdict()
    return {key: int(value) if key == "best_epoch" else float(value) for key, value in values.items()}


def _monitor_process(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
    device: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    peak_rss = 0
    peak_gpu = None
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=dict(env), stdout=stdout, stderr=stderr, text=True)
        watcher = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak_rss = max(peak_rss, watcher.memory_info().rss)
            except psutil.Error:
                pass
            if device == "cuda":
                gpu = _gpu_memory_gb(process.pid)
                if gpu is not None:
                    peak_gpu = max(peak_gpu or 0.0, gpu)
            if time.perf_counter() - started >= timeout_s:
                timed_out = True
                process.kill()
                break
            time.sleep(0.25)
        process.wait()
    return {
        "return_code": process.returncode,
        "time_s": time.perf_counter() - started,
        "cpu_peak_rss_gb": peak_rss / (1024**3),
        "gpu_process_peak_gb": peak_gpu,
        "timed_out": timed_out,
    }


def build_ibot_command(
    *,
    python: Path,
    code_root: Path,
    params: Mapping[str, Any],
    seed: int,
    device: str,
    use_attr: bool,
    log_dir: Path,
    extra_overrides: Mapping[str, Any] | None = None,
) -> list[str]:
    command = [
        str(python),
        str(code_root / "main.py"),
        "--dataset",
        str(params["tuning_dataset_name"]),
        "--seed",
        str(seed),
        "--ib_dim",
        "0",
        "--ib_start_epoch",
        "30",
        "--ib_strength_warmup_epochs",
        "30",
        "--ib_kl_anneal_epochs",
        "50",
        "--no_tensorboard",
        "--dtype",
        "float32",
        "--eval_every",
        "1",
        "--ot_update_every",
        "1",
        "--in_iter",
        "5",
        "--out_iter",
        "10",
        "--sort_metrics",
        "--metric_batch_size",
        "256",
        "--split_protocol",
        "train10_val10_test80",
        "--val_metric",
        "mrr",
        "--log_dir",
        str(log_dir),
    ]
    if device == "cuda":
        command.append("--gpu")
    if use_attr:
        command.append("--use_attr")
    for key in PARAMETER_KEYS:
        command.extend((f"--{key}", _stringify(params[key])))
        command.extend(("--override", f"{key}={_stringify(params[key])}"))
    for key, value in (extra_overrides or {}).items():
        command.extend(("--override", f"{key}={_stringify(value)}"))
    return command


def run_ibot_seed(
    *,
    dataset: str,
    seed: int,
    params: Mapping[str, Any],
    extra_overrides: Mapping[str, Any],
    python: Path,
    code_root: Path,
    data_root: Path,
    task_root: Path,
    device: str,
    timeout_s: float,
) -> dict[str, Any]:
    raw_dataset = load_dataset(data_root, dataset)
    task_root.mkdir(parents=True, exist_ok=True)
    stdout_path = task_root / "stdout.log"
    stderr_path = task_root / "stderr.log"
    command = build_ibot_command(
        python=python,
        code_root=code_root,
        params=params,
        seed=seed,
        device=device,
        use_attr=raw_dataset.x1 is not None and raw_dataset.x2 is not None,
        log_dir=task_root / "tensorboard",
        extra_overrides=extra_overrides,
    )
    prefix = str(raw_dataset.path)[: -len("_0.2.npz")]
    cache_root = runtime_cache_root(task_root, dataset=dataset, seed=seed)
    cache_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": str(seed),
            "PYTHONPYCACHEPREFIX": str(cache_root / "pycache"),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "IBOT_NA_DATASET_PREFIX_OVERRIDE": prefix,
            "IBOT_NA_RWR_CACHE_ROOT": str(cache_root / "rwr"),
        }
    )
    process_info = _monitor_process(
        command,
        cwd=code_root,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_s=timeout_s,
        device=device,
    )
    row: dict[str, Any] = {
        "dataset": dataset,
        "model": "IBOT-NA",
        "seed": seed,
        "status": "OK",
        "time_s": process_info["time_s"],
        "memory_gb": process_info["gpu_process_peak_gb"]
        if device == "cuda" and process_info["gpu_process_peak_gb"] is not None
        else process_info["cpu_peak_rss_gb"],
        "cpu_peak_rss_gb": process_info["cpu_peak_rss_gb"],
        "gpu_process_peak_gb": process_info["gpu_process_peak_gb"],
        "return_code": process_info["return_code"],
        "params": {**dict(params), **dict(extra_overrides)},
        "selection": "best_by_validation_mrr",
        "split_protocol": "train10_val10_test80",
        "command": command,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "runtime_cache_root": str(cache_root),
    }
    if process_info["timed_out"]:
        row.update({"status": "OOM", "error": f"OOM: timeout after {timeout_s:g}s"})
    elif process_info["return_code"] != 0:
        row["status"], row["error"] = _classify_failure(
            stdout_path, stderr_path, process_info["return_code"]
        )
    else:
        try:
            row.update(_parse_ibot_metrics(stdout_path))
        except ValueError as error:
            row.update({"status": "FAILED", "error": str(error)})
    return row
