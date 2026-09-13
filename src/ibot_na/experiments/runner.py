"""Seed-group execution policy for IBOT-NA."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def is_oom_status(row: dict[str, Any]) -> bool:
    status = str(row.get("status", "")).strip().casefold()
    error = str(row.get("error", row.get("failure_reason", ""))).casefold()
    return (
        status in {"oom", "timeout"}
        or "out of memory" in error
        or "timeout" in error
        or "exceeded 3600" in error
    )


def execute_seed_group(
    *,
    dataset: str,
    model: str,
    seeds: Iterable[int],
    run_seed: Callable[[int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run seeds sequentially and stop the group after OOM or timeout."""
    seed_list = [int(seed) for seed in seeds]
    rows: list[dict[str, Any]] = []
    oom_seed = None
    for seed in seed_list:
        if oom_seed is not None:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "status": "SKIPPED_OOM",
                    "error": f"Skipped because seed {oom_seed} ended in OOM or timeout",
                }
            )
            continue
        row = dict(run_seed(seed))
        row.update({"dataset": dataset, "model": model, "seed": seed})
        if is_oom_status(row):
            row["status"] = "OOM"
            oom_seed = seed
        rows.append(row)
    return rows
