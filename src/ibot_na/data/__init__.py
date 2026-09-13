"""Data loading and split interfaces for IBOT-NA."""

from .dataset import AlignmentDataset, load_dataset

__all__ = [
    "AlignmentDataset",
    "load_dataset",
]
