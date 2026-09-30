"""Deterministic split assignment for an opaque image or source-group key."""

from __future__ import annotations

from . import SPLIT_SALT
from .common import sha256_text

SELECTION = "selection"
LOCKED_TEST = "locked_test"


def assign_split(split_key: str) -> str:
    """~20% of source groups land in locked_test for a stable split key."""
    digest = int(sha256_text(f"{SPLIT_SALT}|{split_key}"), 16)
    return LOCKED_TEST if digest % 5 == 0 else SELECTION
