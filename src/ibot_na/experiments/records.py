"""Append-only experiment records and JSON snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def latest_records(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    latest = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["model"]), int(row["seed"]))
        latest[key] = row
    return latest


def append_records(jsonl_path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        return
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    snapshot = read_jsonl(jsonl_path)
    json_path = jsonl_path.with_suffix(".json")
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
