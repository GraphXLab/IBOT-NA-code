"""Load paired-network datasets from the package NPZ format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "edge_index1",
    "edge_index2",
    "edge_attr1",
    "edge_attr2",
    "pos_pairs",
    "test_pairs",
    "num_nodes1",
    "num_nodes2",
}


def _edge_index(value: np.ndarray, key: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.int64)
    if value.ndim != 2:
        raise ValueError(f"{key} must be a two-dimensional array")
    if value.shape[0] == 2:
        return value
    if value.shape[1] == 2:
        return value.T
    raise ValueError(f"{key} must have shape (2, E) or (E, 2)")


def _pairs(value: np.ndarray, key: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"{key} must have shape (N, 2)")
    return value


def _features(value: np.ndarray | None, rows: int, key: str) -> np.ndarray | None:
    if value is None:
        return None
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[0] != rows:
        raise ValueError(f"{key} must have shape ({rows}, D)")
    return value if value.shape[1] else None


def _pair_set(value: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(row) for row in value.tolist()}


@dataclass(frozen=True)
class AlignmentDataset:
    """Two graphs and their fixed supervised/test alignment pools."""

    name: str
    edge_index1: np.ndarray
    edge_index2: np.ndarray
    edge_attr1: np.ndarray | None
    edge_attr2: np.ndarray | None
    x1: np.ndarray | None
    x2: np.ndarray | None
    pos_pairs: np.ndarray
    test_pairs: np.ndarray
    num_nodes1: int
    num_nodes2: int
    path: Path

    def validate(self) -> None:
        if self.edge_index1.size and self.edge_index1.max() >= self.num_nodes1:
            raise ValueError(f"{self.name}: graph 1 edge index exceeds num_nodes1")
        if self.edge_index2.size and self.edge_index2.max() >= self.num_nodes2:
            raise ValueError(f"{self.name}: graph 2 edge index exceeds num_nodes2")
        all_pairs = np.concatenate((self.pos_pairs, self.test_pairs), axis=0)
        if all_pairs.size:
            if all_pairs[:, 0].min() < 0 or all_pairs[:, 0].max() >= self.num_nodes1:
                raise ValueError(f"{self.name}: graph 1 pair index is out of range")
            if all_pairs[:, 1].min() < 0 or all_pairs[:, 1].max() >= self.num_nodes2:
                raise ValueError(f"{self.name}: graph 2 pair index is out of range")
        overlap = _pair_set(self.pos_pairs) & _pair_set(self.test_pairs)
        if overlap:
            raise ValueError(f"{self.name}: supervised/test leakage ({len(overlap)} pairs)")


def available_datasets(root: str | Path) -> list[str]:
    """Return canonical dataset names available below a processed-data root."""
    root = Path(root)
    suffix = "-pa_0.2.npz"
    return sorted(path.name[: -len(suffix)] for path in root.glob(f"*{suffix}"))


def _resolve_path(root: Path, name: str) -> Path:
    requested = name.casefold()
    requested = requested.removesuffix("-pa").removesuffix("_0.2")
    matches = [
        path
        for path in root.glob("*-pa_0.2.npz")
        if path.name[: -len("-pa_0.2.npz")].casefold() == requested
    ]
    if len(matches) != 1:
        choices = ", ".join(available_datasets(root))
        raise FileNotFoundError(f"Unknown dataset {name!r}. Available datasets: {choices}")
    return matches[0]


def load_dataset(root: str | Path, name: str) -> AlignmentDataset:
    """Load and validate one common-format dataset by name."""
    path = _resolve_path(Path(root), name)
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_KEYS - set(data.files)
        if missing:
            raise ValueError(f"{path.name} is missing keys: {sorted(missing)}")

        edge_index1 = _edge_index(data["edge_index1"], "edge_index1")
        edge_index2 = _edge_index(data["edge_index2"], "edge_index2")
        num_nodes1 = int(np.asarray(data["num_nodes1"]).item())
        num_nodes2 = int(np.asarray(data["num_nodes2"]).item())
        dataset = AlignmentDataset(
            name=path.name[: -len("-pa_0.2.npz")],
            edge_index1=edge_index1,
            edge_index2=edge_index2,
            edge_attr1=_features(data["edge_attr1"], edge_index1.shape[1], "edge_attr1"),
            edge_attr2=_features(data["edge_attr2"], edge_index2.shape[1], "edge_attr2"),
            x1=_features(data["x1"] if "x1" in data.files else None, num_nodes1, "x1"),
            x2=_features(data["x2"] if "x2" in data.files else None, num_nodes2, "x2"),
            pos_pairs=_pairs(data["pos_pairs"], "pos_pairs"),
            test_pairs=_pairs(data["test_pairs"], "test_pairs"),
            num_nodes1=num_nodes1,
            num_nodes2=num_nodes2,
            path=path,
        )
    dataset.validate()
    return dataset
