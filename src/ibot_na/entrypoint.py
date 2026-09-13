"""Helpers shared by command-line and IDE script entry points."""

from __future__ import annotations

from collections.abc import Callable


def run_entrypoint(callback: Callable[[], int | None]) -> None:
    """Run a script callback without turning successful IDE runs into errors."""
    status = callback()
    if status not in (None, 0):
        raise SystemExit(status)
